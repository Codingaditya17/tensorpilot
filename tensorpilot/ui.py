"""Self-contained browser dashboard served by the local inference process."""

DASHBOARD_HTML = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>TensorPilot</title>
  <style>
    :root{color-scheme:dark;--bg:#080b10;--panel:#10151d;--line:#26303d;--muted:#91a0b2;--text:#eef4fb;--cyan:#54d7df;--lime:#a8e063;--red:#ff7785}
    *{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 80% -10%,#163847 0,transparent 35%),var(--bg);color:var(--text);font:15px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;min-height:100vh}
    main{max-width:1120px;margin:auto;padding:28px 22px 60px}.top{display:flex;justify-content:space-between;align-items:center;gap:20px;margin-bottom:28px}.brand{font-size:22px;letter-spacing:.12em;font-weight:800}.brand b{color:var(--cyan)}
    .status{display:flex;align-items:center;gap:8px;color:var(--muted)}.dot{width:9px;height:9px;border-radius:50%;background:#6c7784}.dot.ok{background:var(--lime);box-shadow:0 0 12px var(--lime)}.dot.bad{background:var(--red)}
    .hero{display:grid;grid-template-columns:1.35fr .65fr;gap:18px}.panel{background:linear-gradient(145deg,rgba(18,25,35,.96),rgba(11,15,21,.96));border:1px solid var(--line);border-radius:14px;padding:20px;box-shadow:0 18px 60px #0007}
    .eyebrow{color:var(--cyan);font-size:11px;text-transform:uppercase;letter-spacing:.16em}h1{font-family:system-ui,sans-serif;font-size:clamp(30px,5vw,54px);line-height:1.04;margin:10px 0 14px;letter-spacing:-.035em}h1 span{color:var(--cyan)}p{color:var(--muted);max-width:650px}
    .stats{display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin-top:22px}.metric{padding:14px;border:1px solid var(--line);border-radius:10px;background:#0b1017}.metric strong{display:block;font:700 21px system-ui,sans-serif}.metric small{color:var(--muted)}
    label{display:block;color:var(--muted);font-size:12px;margin:13px 0 7px}textarea,input{width:100%;background:#080c12;border:1px solid var(--line);color:var(--text);border-radius:9px;padding:12px;font:inherit;outline:none}textarea:focus,input:focus{border-color:var(--cyan)}textarea{min-height:130px;resize:vertical}
    .row{display:flex;gap:10px}.row>*{flex:1}button{margin-top:16px;border:0;border-radius:9px;background:var(--cyan);color:#031014;padding:12px 18px;font:800 14px system-ui,sans-serif;cursor:pointer;width:100%}button:disabled{opacity:.45;cursor:wait}.output{white-space:pre-wrap;min-height:150px;max-height:360px;overflow:auto;background:#070b10;border:1px solid var(--line);border-radius:10px;padding:16px;color:#d9e7f3;margin-top:16px}.meta{font-size:11px;color:var(--muted);margin-top:10px;min-height:18px}
    .foot{margin-top:18px;color:#617082;font-size:11px}.error{color:var(--red)}
    @media(max-width:780px){.hero{grid-template-columns:1fr}.top{align-items:flex-start}.stats{grid-template-columns:1fr 1fr}main{padding:20px 14px}}
  </style>
</head>
<body><main>
  <div class="top"><div class="brand">TENSOR<b>PILOT</b></div><div class="status"><i id="dot" class="dot"></i><span id="status">connecting</span></div></div>
  <div class="hero">
    <section class="panel">
      <div class="eyebrow">Local MLX inference runtime</div>
      <h1>Drive the model.<br><span>Measure everything.</span></h1>
      <p>Bounded admission, streaming generation, continuous batching, cancellation, deadlines, and real Apple GPU measurements in one local service.</p>
      <div class="stats">
        <div class="metric"><strong id="mode">—</strong><small>execution mode</small></div>
        <div class="metric"><strong id="active">—</strong><small>active / capacity</small></div>
        <div class="metric"><strong id="queued">—</strong><small>queued requests</small></div>
        <div class="metric"><strong id="device">—</strong><small>compute device</small></div>
      </div>
      <div class="foot" id="model">Loading model metadata…</div>
    </section>
    <section class="panel">
      <div class="eyebrow">Inference console</div>
      <label for="prompt">Prompt</label>
      <textarea id="prompt">Explain continuous batching in two short sentences.</textarea>
      <div class="row"><div><label for="tokens">Max tokens</label><input id="tokens" type="number" min="1" max="256" value="64"></div><div><label for="timeout">Timeout (s)</label><input id="timeout" type="number" min="1" max="120" value="60"></div></div>
      <label for="key">API key (only if configured)</label><input id="key" type="password" autocomplete="off" placeholder="optional">
      <button id="run">Generate</button>
      <div id="output" class="output">Output will stream here.</div>
      <div id="meta" class="meta"></div>
    </section>
  </div>
</main>
<script>
const $=id=>document.getElementById(id), auth=()=>$('key').value?{Authorization:'Bearer '+$('key').value}:{};
async function refresh(){try{const r=await fetch('/health',{headers:auth()}),h=await r.json();if(!r.ok)throw Error(h.error?.message||'HTTP '+r.status);$('dot').className='dot '+(h.ready?'ok':'bad');$('status').textContent=h.ready?'ready':'unavailable';$('mode').textContent=h.execution_mode||'serial';$('active').textContent=`${h.active_sequences||0} / ${h.execution_concurrency||1}`;$('queued').textContent=h.queue_depth??'—';$('device').textContent=h.mlx_device?.device_name||h.backend||'—';$('model').textContent=`${h.model||'model'} · v${h.service_version||'?'} · ${h.scheduler||'fifo'} scheduler`; }catch(e){$('dot').className='dot bad';$('status').textContent='offline';$('meta').textContent=e.message;$('meta').className='meta error'}}
async function generate(){const btn=$('run'),out=$('output'),meta=$('meta');btn.disabled=true;out.textContent='';meta.textContent='Starting…';meta.className='meta';const started=performance.now();try{const r=await fetch('/v1/generate',{method:'POST',headers:{'Content-Type':'application/json',...auth()},body:JSON.stringify({prompt:$('prompt').value,max_tokens:Number($('tokens').value),timeout_s:Number($('timeout').value)})});if(!r.ok){const b=await r.json();throw Error(b.error?.message||'HTTP '+r.status)}const reader=r.body.getReader(),decoder=new TextDecoder();let buf='',count=0,doneEvent;while(true){const x=await reader.read();if(x.done)break;buf+=decoder.decode(x.value,{stream:true});const lines=buf.split('\n');buf=lines.pop();for(const line of lines){if(!line)continue;const e=JSON.parse(line);if(e.type==='token'&&e.output){out.textContent+=e.text;count++}else if(e.type==='error')throw Error(e.error||e.code);else if(e.type==='done')doneEvent=e;out.scrollTop=out.scrollHeight}}const elapsed=(performance.now()-started)/1000;meta.textContent=`${count} tokens · ${elapsed.toFixed(2)}s client · ${(doneEvent?.queue_wait_s||0).toFixed(3)}s queue · ${doneEvent?.execution_mode||'serial'}`;}catch(e){out.textContent=out.textContent||'Generation failed.';meta.textContent=e.message;meta.className='meta error'}finally{btn.disabled=false;refresh()}}
$('run').addEventListener('click',generate);$('key').addEventListener('change',refresh);refresh();setInterval(refresh,3000);
</script></body></html>'''
