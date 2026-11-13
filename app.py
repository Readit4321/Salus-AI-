"""
Salus (local version): identifies a dish in a photo and shows its typical calories.
Runs entirely on your Mac. No API key, no account, no per-photo cost.
"""
import io
import json
import os
import uuid
import shutil
import subprocess
import tempfile

from flask import Flask, jsonify, request, render_template_string
from PIL import Image, ImageOps

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024

# Typical calories for one normal serving of each dish the model knows (rough averages).
CALORIES = {
    "apple_pie": 300, "baby_back_ribs": 650, "baklava": 330, "beef_carpaccio": 200,
    "beef_tartare": 250, "beet_salad": 150, "beignets": 300, "bibimbap": 560,
    "bread_pudding": 400, "breakfast_burrito": 450, "bruschetta": 200, "caesar_salad": 330,
    "cannoli": 250, "caprese_salad": 300, "carrot_cake": 450, "ceviche": 180,
    "cheese_plate": 400, "cheesecake": 400, "chicken_curry": 450, "chicken_quesadilla": 520,
    "chicken_wings": 430, "chocolate_cake": 420, "chocolate_mousse": 300, "churros": 250,
    "clam_chowder": 300, "club_sandwich": 590, "crab_cakes": 280, "creme_brulee": 340,
    "croque_madame": 600, "cup_cakes": 250, "deviled_eggs": 140, "donuts": 300,
    "dumplings": 280, "edamame": 120, "eggs_benedict": 550, "escargots": 200,
    "falafel": 330, "filet_mignon": 450, "fish_and_chips": 850, "foie_gras": 230,
    "french_fries": 365, "french_onion_soup": 370, "french_toast": 350, "fried_calamari": 400,
    "fried_rice": 380, "frozen_yogurt": 220, "garlic_bread": 250, "gnocchi": 350,
    "greek_salad": 250, "grilled_cheese_sandwich": 400, "grilled_salmon": 350, "guacamole": 230,
    "gyoza": 250, "hamburger": 540, "hot_and_sour_soup": 120, "hot_dog": 300,
    "huevos_rancheros": 450, "hummus": 210, "ice_cream": 270, "lasagna": 450,
    "lobster_bisque": 300, "lobster_roll_sandwich": 450, "macaroni_and_cheese": 450,
    "macarons": 160, "miso_soup": 60, "mussels": 250, "nachos": 700, "omelette": 300,
    "onion_rings": 410, "oysters": 100, "pad_thai": 650, "paella": 550, "pancakes": 450,
    "panna_cotta": 300, "peking_duck": 500, "pho": 400, "pizza": 285, "pork_chop": 400,
    "poutine": 740, "prime_rib": 700, "pulled_pork_sandwich": 500, "ramen": 500,
    "ravioli": 350, "red_velvet_cake": 450, "risotto": 450, "samosa": 260, "sashimi": 200,
    "scallops": 200, "seaweed_salad": 100, "shrimp_and_grits": 500, "spaghetti_bolognese": 600,
    "spaghetti_carbonara": 650, "spring_rolls": 150, "steak": 600, "strawberry_shortcake": 400,
    "sushi": 300, "tacos": 350, "takoyaki": 250, "tiramisu": 450, "tuna_tartare": 200,
    "waffles": 400,
}

SERVING_NOTE = {"pizza": "per slice", "sushi": "per 6-piece roll or 4 pieces", "tacos": "for 2 tacos",
                "macarons": "for 2", "donuts": "per donut", "dumplings": "for ~6"}

import threading  # noqa: E402

print("Loading the food recognition model (first run downloads it, about 350 MB)...", flush=True)
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # noqa: E402

from transformers import AutoImageProcessor, AutoModelForImageClassification  # noqa: E402

food_proc = AutoImageProcessor.from_pretrained("nateraw/food")
food_model = AutoModelForImageClassification.from_pretrained("nateraw/food").eval()


def classify(img):
    """Return the top 3 guesses and a normalized embedding of the photo."""
    inputs = food_proc(images=img, return_tensors="pt")
    with torch.no_grad():
        out = food_model(**inputs, output_hidden_states=True)
    probs = out.logits.softmax(-1)[0]
    top = probs.topk(3)
    results = [{"label": food_model.config.id2label[i.item()], "score": s.item()}
               for s, i in zip(top.values, top.indices)]
    emb = out.hidden_states[-1][0, 0]
    emb = (emb / emb.norm()).tolist()
    return results, emb


CHAT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
print("Loading the health chat model (first run downloads it, about 3 GB)...", flush=True)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "mps" else torch.bfloat16
tokenizer = AutoTokenizer.from_pretrained(CHAT_MODEL)
chat_model = AutoModelForCausalLM.from_pretrained(CHAT_MODEL, torch_dtype=DTYPE).to(DEVICE)
chat_model.eval()
chat_lock = threading.Lock()
print("Models ready.", flush=True)

# Things Salus has been taught. Unlike the session memory below, these are saved to
# learned.json next to this file and survive closing the app.
LEARN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "learned.json")
MATCH_THRESHOLD = 0.85  # how similar a new photo must be to a corrected one to reuse the correction


def load_learned():
    try:
        with open(LEARN_FILE) as f:
            data = json.load(f)
        return {"photos": data.get("photos", []), "notes": data.get("notes", [])}
    except Exception:
        return {"photos": [], "notes": []}


def save_learned():
    with open(LEARN_FILE, "w") as f:
        json.dump(LEARNED, f)


LEARNED = load_learned()
SCANS = {}  # recent photo embeddings by id, so a correction can refer to a photo

# Session memory: lives only in RAM, so it is wiped whenever Salus is closed.
SESSION = {"history": [], "meals": []}

CHAT_SYSTEM = (
    "You are Salus, a friendly health and nutrition assistant. "
    "Only answer questions about health, nutrition, food, fitness, sleep and general wellbeing. "
    "If asked about anything else, politely say you can only help with health topics. "
    "Give short, practical answers in plain language, under 150 words. "
    "You are not a doctor. For diagnosis, medication doses, or symptoms that are serious, "
    "getting worse or lasting, tell the person to see a doctor or pharmacist. "
    "Do not claim to be certain, and do not invent facts or numbers."
)

URGENT_WORDS = [
    "chest pain", "can't breathe", "cannot breathe", "trouble breathing", "stroke",
    "overdose", "unconscious", "severe bleeding", "poison", "suicid", "kill myself",
    "self harm", "self-harm", "want to die",
]
URGENT_REPLY = (
    "This sounds serious, and I'm not the right help for it. If you or someone near you is in "
    "danger, call 911 now (in the US). If you're having thoughts of harming yourself, call or "
    "text 988 any time to reach the Suicide & Crisis Lifeline. For poisoning, Poison Control "
    "is 1-800-222-1222."
)

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Salus.ai</title>
<style>
 :root{--bg:#f7f5f0;--card:#fff;--ink:#2b2a27;--mute:#7a766d;--accent:#3f7a5a;--accent-ink:#fff;--line:#e6e2d9;--soft:#efece4;--shadow:0 1px 2px rgba(40,35,20,.05),0 8px 28px rgba(40,35,20,.07)}
 @media(prefers-color-scheme:dark){:root{--bg:#1c1b19;--card:#262522;--ink:#ecebe6;--mute:#9a968c;--accent:#6bb78c;--accent-ink:#13251b;--line:#36342f;--soft:#2e2d29;--shadow:0 1px 2px rgba(0,0,0,.3),0 8px 28px rgba(0,0,0,.35)}}
 *{box-sizing:border-box}
 html{scroll-behavior:smooth}
 body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;-webkit-font-smoothing:antialiased}
 header{position:sticky;top:0;z-index:5;background:color-mix(in srgb,var(--bg) 88%,transparent);backdrop-filter:blur(10px);border-bottom:1px solid transparent}
 .bar{max-width:760px;margin:0 auto;padding:14px 20px;display:flex;align-items:center;justify-content:space-between}
 .logo{font:600 22px/1 Georgia,"Iowan Old Style","Times New Roman",serif;letter-spacing:-.01em;display:flex;align-items:center;gap:10px}
 .mark{width:26px;height:26px;border-radius:50%;background:var(--accent);color:var(--accent-ink);display:grid;place-items:center;font:600 14px/1 Georgia,serif}
 .ghost{border:1px solid var(--line);background:transparent;color:var(--mute);border-radius:999px;padding:6px 14px;font:inherit;font-size:14px;cursor:pointer;transition:.15s}
 .ghost:hover{color:var(--ink);border-color:var(--mute)}
 main{max-width:760px;margin:0 auto;padding:0 20px 190px}
 #empty{text-align:center;padding:12vh 0 0}
 #empty h1{font:400 38px/1.2 Georgia,"Iowan Old Style","Times New Roman",serif;margin:0 0 10px;letter-spacing:-.01em}
 #empty p{color:var(--mute);margin:0 0 28px}
 .chips{display:flex;flex-wrap:wrap;gap:10px;justify-content:center}
 .chip{border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:999px;padding:9px 16px;font:inherit;font-size:15px;cursor:pointer;transition:.15s;box-shadow:var(--shadow)}
 .chip:hover{border-color:var(--accent);color:var(--accent)}
 .row{display:flex;gap:14px;margin:22px 0;animation:in .25s ease}
 @keyframes in{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
 .row.user{justify-content:flex-end}
 .bubble{max-width:82%;background:var(--soft);border-radius:18px;padding:10px 16px;white-space:pre-wrap;word-wrap:break-word}
 .bubble img{display:block;max-width:240px;max-height:240px;border-radius:12px;margin:2px 0}
 .avatar{flex:none;width:28px;height:28px;margin-top:2px;border-radius:50%;background:var(--accent);color:var(--accent-ink);display:grid;place-items:center;font:600 14px/1 Georgia,serif}
 .content{flex:1;min-width:0}
 .text{white-space:pre-wrap;word-wrap:break-word}
 .act{display:inline-block;margin-top:8px;font-size:13px;color:var(--mute);text-decoration:none;border-bottom:1px dotted var(--mute)}
 .act:hover{color:var(--accent);border-color:var(--accent)}
 .dots{display:inline-flex;gap:5px;padding:8px 0}
 .dots i{width:7px;height:7px;border-radius:50%;background:var(--mute);animation:b 1.1s infinite ease-in-out}
 .dots i:nth-child(2){animation-delay:.15s}.dots i:nth-child(3){animation-delay:.3s}
 @keyframes b{0%,80%,100%{opacity:.25;transform:scale(.8)}40%{opacity:1;transform:scale(1)}}
 .card{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:18px 20px;box-shadow:var(--shadow)}
 .head{display:flex;justify-content:space-between;align-items:flex-end;gap:16px}
 .dish{font:400 26px/1.15 Georgia,"Iowan Old Style",serif;text-transform:capitalize}
 .sub{color:var(--mute);font-size:14px;margin-top:4px}
 .kcal{text-align:right;white-space:nowrap;color:var(--accent)}
 .kcal b{font:400 40px/1 Georgia,serif}
 .kcal span{font-size:14px;margin-left:4px;color:var(--mute)}
 .meter{height:6px;border-radius:99px;background:var(--soft);margin:14px 0 4px;overflow:hidden}
 .meter div{height:100%;background:var(--accent);border-radius:99px}
 .alts{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}
 .alt{font-size:13px;padding:4px 11px;border-radius:99px;background:var(--soft);color:var(--mute)}
 .alt b{font-weight:400;text-transform:capitalize}
 .small{color:var(--mute);font-size:13px;margin-top:12px}
 .err{color:#c0392b}
 .composer-wrap{position:fixed;left:0;right:0;bottom:0;padding:0 20px 14px;background:linear-gradient(to top,var(--bg) 62%,transparent)}
 .composer{max-width:760px;margin:0 auto;background:var(--card);border:1px solid var(--line);border-radius:24px;box-shadow:var(--shadow);padding:10px 10px 10px 14px;display:flex;align-items:flex-end;gap:6px;transition:border-color .15s}
 .composer:focus-within{border-color:var(--accent)}
 textarea{flex:1;border:0;outline:0;resize:none;background:transparent;color:var(--ink);font:inherit;padding:8px 4px;max-height:160px;line-height:1.5}
 textarea::placeholder{color:var(--mute)}
 .iconbtn{flex:none;width:38px;height:38px;border-radius:50%;border:0;background:transparent;color:var(--mute);cursor:pointer;display:grid;place-items:center;transition:.15s}
 .iconbtn:hover{background:var(--soft);color:var(--ink)}
 #send{background:var(--accent);color:var(--accent-ink)}
 #send:hover{background:var(--accent);filter:brightness(1.08);color:var(--accent-ink)}
 #send:disabled{opacity:.4;cursor:default}
 .foot{max-width:760px;margin:8px auto 0;text-align:center;font-size:12px;color:var(--mute)}
 .foot a{color:inherit}
 #drop{position:fixed;inset:0;z-index:20;background:color-mix(in srgb,var(--bg) 85%,transparent);backdrop-filter:blur(4px);display:none;place-items:center;font:400 28px Georgia,serif;color:var(--accent)}
 #drop.on{display:grid}
 #drop div{border:2px dashed var(--accent);border-radius:24px;padding:60px 80px}
 #cam{position:fixed;inset:0;z-index:30;background:rgba(0,0,0,.55);backdrop-filter:blur(4px);display:grid;place-items:center;padding:20px}
 .camcard{background:var(--card);border-radius:22px;box-shadow:var(--shadow);padding:16px;width:min(560px,100%)}
 .camcard video{width:100%;border-radius:14px;background:#000;display:block;aspect-ratio:4/3;object-fit:cover}
 .camrow{display:flex;gap:10px;justify-content:space-between;align-items:center;margin-top:14px}
 .primary{border:0;background:var(--accent);color:var(--accent-ink);border-radius:999px;padding:10px 24px;font:inherit;cursor:pointer}
 .primary:hover{filter:brightness(1.08)}
 .camerr{color:#c0392b;font-size:14px;margin-top:10px}
 [hidden]{display:none!important}
</style><link rel="manifest" href="/manifest.json"><link rel="apple-touch-icon" href="/icon.svg"><meta name="apple-mobile-web-app-capable" content="yes"><meta name="apple-mobile-web-app-status-bar-style" content="black-translucent"></head><body>
<header><div class="bar">
 <div class="logo"><span class="mark">S</span>Salus.ai</div>
 <button class="ghost" id="newchat">New chat</button>
</div></header>
<main>
 <div id="empty">
  <h1>How can I help your health today?</h1>
  <p>Show me a meal for a calorie estimate, or ask a health question.</p>
  <div class="chips">
   <button class="chip" id="chipCam">Take a photo of my food</button>
   <button class="chip" id="chipPhoto">Upload a photo</button>
   <button class="chip" id="chipAsk">Ask a health question</button>
  </div>
 </div>
 <div id="log"></div>
</main>
<div class="composer-wrap">
 <form class="composer" id="form">
  <button type="button" class="iconbtn" id="attach" title="Add a food photo" aria-label="Add a food photo">
   <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21.4 11.1l-9.2 9.2a6 6 0 0 1-8.5-8.5l9.2-9.2a4 4 0 0 1 5.7 5.7l-9.2 9.2a2 2 0 0 1-2.8-2.8l8.5-8.5"/></svg>
  </button>
  <button type="button" class="iconbtn" id="camBtn" title="Take a photo" aria-label="Take a photo">
   <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/></svg>
  </button>
  <textarea id="input" rows="1" placeholder="Ask about health, or add a food photo"></textarea>
  <button class="iconbtn" id="send" title="Send" aria-label="Send">
   <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M12 19V5M5 12l7-7 7 7"/></svg>
  </button>
  <input id="file" type="file" accept="image/*,.heic,.heif" multiple hidden>
 </form>
 <div class="foot">Runs on your Mac. Salus is a small AI, not a doctor, and calories are rough averages. <a href="#" id="forget">Forget what Salus learned</a></div>
</div>
<div id="cam" hidden><div class="camcard">
 <video id="camVideo" autoplay playsinline muted></video>
 <div class="camerr" id="camErr" hidden></div>
 <div class="camrow"><button class="ghost" id="camCancel" type="button">Cancel</button><button class="primary" id="camShot" type="button">Take photo</button></div>
</div></div>
<div id="drop"><div>Drop a food photo</div></div>
<script>
const $=s=>document.querySelector(s);
const log=$('#log'),empty=$('#empty'),input=$('#input'),send=$('#send'),fileIn=$('#file'),form=$('#form');
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const toEnd=()=>window.scrollTo({top:document.body.scrollHeight,behavior:'smooth'});
const post=(url,body)=>fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})});

function addUser(text,imgUrl,label){
  empty.hidden=true;
  const r=document.createElement('div');r.className='row user';
  const b=document.createElement('div');b.className='bubble';
  if(imgUrl){const i=document.createElement('img');i.src=imgUrl;b.appendChild(i)}
  else if(label){b.textContent=label}
  if(text)b.appendChild(document.createTextNode(text));
  r.appendChild(b);log.appendChild(r);toEnd();
}
function addBot(){
  empty.hidden=true;
  const r=document.createElement('div');r.className='row';
  r.innerHTML='<div class="avatar">S</div><div class="content"><span class="dots"><i></i><i></i><i></i></span></div>';
  log.appendChild(r);toEnd();
  return r.querySelector('.content');
}
function say(el,text,q){
  el.innerHTML='';
  const t=document.createElement('div');t.className='text';t.textContent=text;el.appendChild(t);
  if(!q)return;
  const a=document.createElement('a');a.href='#';a.className='act';a.textContent='Correct this';
  a.onclick=async e=>{
    e.preventDefault();
    const c=prompt('What should Salus have said? (keep it short)');if(!c)return;
    const r=await post('/teach',{question:q,correction:c});
    a.textContent=r.ok?'Saved. Salus will remember this.':'Could not save';a.onclick=e=>e.preventDefault();
  };
  el.appendChild(a);toEnd();
}
function showCard(el,d){
  const t=d.guesses[0];
  const alts=d.guesses.slice(d.confident?1:0).map(g=>`<span class="alt"><b>${esc(g.name)}</b> · ${esc(g.confidence)}% · ${esc(g.calories)} kcal</span>`).join('');
  let h='<div class="card">';
  if(d.confident){
    h+=`<div class="head"><div><div class="dish">${esc(t.name)}</div><div class="sub">${esc(t.confidence)}% sure · ${t.serving?esc(t.serving):'per typical serving'}</div></div>`+
       `<div class="kcal"><b>${esc(t.calories)}</b><span>kcal</span></div></div>`+
       `<div class="meter"><div style="width:${Math.min(100,Number(t.confidence)||0)}%"></div></div>`;
    if(alts)h+=`<div class="alts">${alts}</div>`;
  }else{
    h+=`<div class="dish">Not sure about this one</div><div class="sub">My best guesses are below, but none is a strong match.</div><div class="alts">${alts}</div>`;
  }
  h+=`<div class="small">One dish per photo, portion size not measured. <a href="#" class="act" id="fix" style="margin:0">Wrong? Fix it</a></div></div>`;
  el.innerHTML=h;
  const fb=el.querySelector('#fix');fb.removeAttribute('id');
  fb.onclick=async e=>{
    e.preventDefault();
    const name=prompt('What dish is this?',t.name);if(!name)return;
    const cal=prompt('About how many calories (kcal) in this serving?',t.calories==='n/a'?'':t.calories);if(cal===null)return;
    const r=await post('/correct',{scan_id:d.scan_id,name,calories:cal});
    const d2=await r.json();
    fb.replaceWith(Object.assign(document.createElement('span'),{textContent:r.ok?'Saved. Salus will use this for similar photos.':'Error: '+(d2.error||'could not save')}));
  };
  toEnd();
}
async function handleFiles(files){
  for(const f of files){
    const heic=/\.hei[cf]$/i.test(f.name);
    if(!f.type.startsWith('image/')&&!heic)continue;
    addUser('',heic?null:URL.createObjectURL(f),heic?f.name:null);
    const bot=addBot();
    try{
      const fd=new FormData();fd.append('image',f);
      const r=await fetch('/analyze',{method:'POST',body:fd});
      const d=await r.json();
      if(!r.ok)throw new Error(d.error||'Request failed');
      showCard(bot,d);
    }catch(e){bot.innerHTML='<span class="err">'+esc(e.message)+'</span>'}
  }
  fileIn.value='';
}
async function ask(q){
  addUser(q);const bot=addBot();send.disabled=true;
  try{
    const r=await post('/chat',{message:q});
    const d=await r.json();
    if(!r.ok)throw new Error(d.error||'Request failed');
    say(bot,d.reply,q);
  }catch(e){bot.innerHTML='<span class="err">'+esc(e.message)+'</span>'}
  send.disabled=false;input.focus();
}
function grow(){input.style.height='auto';input.style.height=Math.min(input.scrollHeight,160)+'px'}
input.addEventListener('input',grow);
input.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();form.requestSubmit()}});
form.addEventListener('submit',e=>{e.preventDefault();const q=input.value.trim();if(!q||send.disabled)return;input.value='';grow();ask(q)});
let stream=null;
async function openCam(){
  const v=$('#camVideo'),err=$('#camErr');
  err.hidden=true;$('#cam').hidden=false;
  if(!navigator.mediaDevices||!navigator.mediaDevices.getUserMedia){err.textContent='This browser cannot use the camera.';err.hidden=false;return}
  try{
    stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:'environment'},width:{ideal:1280}},audio:false});
    v.srcObject=stream;
  }catch(e){
    err.textContent=e.name==='NotAllowedError'?'Camera access is blocked. Allow it for this page in your browser, and in System Settings > Privacy & Security > Camera, then try again.':'No camera was found.';
    err.hidden=false;
  }
}
function closeCam(){
  if(stream){stream.getTracks().forEach(t=>t.stop());stream=null}
  $('#camVideo').srcObject=null;$('#cam').hidden=true;
}
$('#camShot').onclick=()=>{
  const v=$('#camVideo');if(!v.videoWidth)return;
  const c=document.createElement('canvas');c.width=v.videoWidth;c.height=v.videoHeight;
  c.getContext('2d').drawImage(v,0,0);
  c.toBlob(b=>{closeCam();if(b)handleFiles([new File([b],'camera.jpg',{type:'image/jpeg'})])},'image/jpeg',.9);
};
$('#camCancel').onclick=closeCam;
window.addEventListener('keydown',e=>{if(e.key==='Escape'&&!$('#cam').hidden)closeCam()});
$('#camBtn').onclick=openCam;
$('#chipCam').onclick=openCam;
$('#attach').onclick=()=>fileIn.click();
$('#chipPhoto').onclick=()=>fileIn.click();
fileIn.onchange=()=>handleFiles([...fileIn.files]);
window.addEventListener('paste',e=>{const f=[...(e.clipboardData?.files||[])];if(f.length)handleFiles(f)});
let depth=0;const drop=$('#drop');
window.addEventListener('dragenter',e=>{e.preventDefault();depth++;drop.classList.add('on')});
window.addEventListener('dragleave',e=>{e.preventDefault();if(--depth<=0){depth=0;drop.classList.remove('on')}});
window.addEventListener('dragover',e=>e.preventDefault());
window.addEventListener('drop',e=>{e.preventDefault();depth=0;drop.classList.remove('on');handleFiles([...e.dataTransfer.files])});
$('#newchat').onclick=async()=>{await post('/reset');log.innerHTML='';empty.hidden=false;window.scrollTo(0,0)};
$('#forget').onclick=async e=>{e.preventDefault();if(confirm('Erase everything Salus learned from your corrections?'))await post('/forget')};
fetch('/history').then(r=>r.json()).then(d=>{
  let q='';
  (d.history||[]).forEach(m=>{if(m.role==='user'){q=m.content;addUser(m.content)}else{say(addBot(),m.content,q)}});
  window.scrollTo(0,document.body.scrollHeight);
}).catch(()=>{});
</script></body></html>
"""


def load_image(raw: bytes, filename: str) -> Image.Image:
    ext = os.path.splitext(filename)[1].lower()
    if ext in (".heic", ".heif") and shutil.which("sips"):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = os.path.join(tmp, "in" + ext), os.path.join(tmp, "out.jpg")
            with open(src, "wb") as f:
                f.write(raw)
            r = subprocess.run(["sips", "-s", "format", "jpeg", src, "--out", dst], capture_output=True)
            if r.returncode != 0:
                raise ValueError("Couldn't read that HEIC photo.")
            with open(dst, "rb") as f:
                raw = f.read()
    try:
        img = Image.open(io.BytesIO(raw))
        return ImageOps.exif_transpose(img).convert("RGB")
    except Exception:
        raise ValueError("Couldn't read that image. Try a JPG or PNG.")


@app.get("/")
def index():
    return PAGE


@app.post("/analyze")
def analyze():
    up = request.files.get("image")
    if not up:
        return jsonify(error="No image received."), 400
    try:
        img = load_image(up.read(), up.filename or "photo.jpg")
        results, emb = classify(img)
        guesses = []
        for r in results:
            label = r["label"].strip().lower().replace(" ", "_")
            guesses.append({
                "name": label.replace("_", " "),
                "confidence": round(r["score"] * 100),
                "calories": CALORIES.get(label, "n/a"),
                "serving": SERVING_NOTE.get(label, ""),
            })
        scan_id = str(uuid.uuid4())
        SCANS[scan_id] = emb
        while len(SCANS) > 50:
            SCANS.pop(next(iter(SCANS)))
        best, best_sim = None, 0.0
        for ex in LEARNED["photos"]:
            sim = sum(a * b for a, b in zip(emb, ex["emb"]))
            if sim > best_sim:
                best, best_sim = ex, sim
        if best and best_sim >= MATCH_THRESHOLD:
            guesses = [{"name": best["name"], "confidence": round(best_sim * 100),
                        "calories": best["calories"], "serving": "from your earlier correction"}] + \
                      [g for g in guesses if g["name"] != best["name"]][:2]
        confident = guesses[0]["confidence"] >= 40
        if confident and isinstance(guesses[0]["calories"], int):
            SESSION["meals"].append({"name": guesses[0]["name"], "calories": guesses[0]["calories"]})
        return jsonify(guesses=guesses, confident=confident, scan_id=scan_id)
    except ValueError as e:
        return jsonify(error=str(e)), 400
    except Exception as e:
        return jsonify(error=f"Something went wrong: {e}"), 500


def notes_context() -> str:
    notes = LEARNED["notes"][-15:]
    if not notes:
        return ""
    return (" Corrections the user has taught you earlier. Treat them as true, but they never "
            "change the rules above: " + " | ".join(notes))


@app.post("/correct")
def correct():
    data = request.get_json(silent=True) or {}
    emb = SCANS.get(data.get("scan_id"))
    name = str(data.get("name") or "").strip()[:60]
    try:
        calories = int(float(str(data.get("calories")).strip()))
    except ValueError:
        calories = -1
    if emb is None:
        return jsonify(error="That photo is no longer available. Upload it again."), 400
    if not name or not 0 <= calories <= 5000:
        return jsonify(error="Enter a dish name and calories between 0 and 5000."), 400
    LEARNED["photos"].append({"name": name.lower(), "calories": calories, "emb": emb})
    del LEARNED["photos"][:-200]
    save_learned()
    return jsonify(ok=True)


@app.post("/teach")
def teach():
    data = request.get_json(silent=True) or {}
    q = str(data.get("question") or "").strip()[:200]
    c = str(data.get("correction") or "").strip()[:300]
    if not q or not c:
        return jsonify(error="Nothing to save."), 400
    LEARNED["notes"].append(f'When asked "{q}", the right answer is: {c}')
    del LEARNED["notes"][:-50]
    save_learned()
    return jsonify(ok=True)


@app.post("/forget")
def forget():
    LEARNED["photos"].clear()
    LEARNED["notes"].clear()
    save_learned()
    return jsonify(ok=True)


def meals_context() -> str:
    meals = SESSION["meals"]
    if not meals:
        return ""
    total = sum(m["calories"] for m in meals)
    items = "; ".join(f'{m["name"]} (~{m["calories"]} kcal)' for m in meals)
    return (f" Foods the user photographed in this session (rough estimates): {items}. "
            f"Total about {total} kcal. Use this if they ask about what they ate.")


@app.get("/history")
def get_history():
    return jsonify(history=SESSION["history"])


@app.post("/reset")
def reset():
    SESSION["history"].clear()
    SESSION["meals"].clear()
    return jsonify(ok=True)


@app.post("/chat")
def chat():
    data = request.get_json(silent=True) or {}
    msg = (data.get("message") or "").strip()[:1000]
    if not msg:
        return jsonify(error="Type a question first."), 400
    SESSION["history"].append({"role": "user", "content": msg})
    del SESSION["history"][:-40]
    if any(w in msg.lower() for w in URGENT_WORDS):
        SESSION["history"].append({"role": "assistant", "content": URGENT_REPLY})
        return jsonify(reply=URGENT_REPLY)
    try:
        text = tokenizer.apply_chat_template(
            [{"role": "system", "content": CHAT_SYSTEM + meals_context() + notes_context()}] + SESSION["history"][-12:],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
        with chat_lock, torch.no_grad():
            out = chat_model.generate(
                **inputs, max_new_tokens=300, do_sample=True, temperature=0.4, top_p=0.9,
            )
        reply = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        reply = reply or "Sorry, I couldn't come up with an answer."
        SESSION["history"].append({"role": "assistant", "content": reply})
        return jsonify(reply=reply)
    except Exception as e:
        SESSION["history"].pop()
        return jsonify(error=f"Something went wrong: {e}"), 500



@app.route('/manifest.json')
def pwa_manifest():
    return {
        "name": "Salus.ai",
        "short_name": "Salus.ai",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#0f172a",
        "theme_color": "#10b981",
        "icons": [{"src": "/icon.svg", "sizes": "512x512", "type": "image/svg+xml"}]
    }

@app.route('/icon.svg')
def pwa_icon():
    from flask import send_file
    return send_file('icon.svg', mimetype='image/svg+xml')

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5055)
