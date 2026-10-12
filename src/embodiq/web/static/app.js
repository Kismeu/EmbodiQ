'use strict';
const $ = id => document.getElementById(id);
const modelNames = {openvla:'OpenVLA', vla_adapter:'VLA-Adapter', fastwam:'FastWAM'};
const methodNames = {fp16:'原始精度', rtn:'四位量化（RTN）', awq:'AWQ · 待接入', hsa:'HSA · 待接入'};
const suiteNames = {libero_spatial:'空间关系', libero_object:'目标物体', libero_goal:'任务目标', libero_10:'长时程操作'};
const suiteDescriptions = {libero_spatial:'理解物体的位置关系',libero_object:'识别并操作指定物体',libero_goal:'根据指令完成目标',libero_10:'连续完成多步操作'};
const phases = {planned:'等待开始',starting:'正在启动',preflight:'检查模型与输入',waiting_for_gpu:'等待空闲显卡',running:'正在评测',complete:'已完成',blocked:'需要处理',failed:'运行失败',stopped:'已停止',interrupted:'运行中断',pending:'等待开始'};
const pageInfo = {create:['新建评测','选择模型、量化方法和任务集合。'],tasks:['任务管理','查看评测进度，管理已提交的实验。'],results:['评测结果','按任务集合比较成功率、响应时间和显存占用。']};
let token='', current='', state=null, allRuns=[], refreshing=false, previewRevision=0, reportRevision=0, submitting=false, reportData=null, reportSuite='all', readingReport=false,reportMode='performance',tradeoffMetric='steady_nvml_gib';
let gpuMemoryRequest=null,gpuMemoryRevision=0,gpuMemorySelection=null;
let gpuListLoading=false,gpuListSelection=null;
function node(tag,text,cls){const el=document.createElement(tag);if(text!==undefined)el.textContent=text;if(cls)el.className=cls;return el;}
function readableError(message){
  const text=String(message);
  if(/gpu_busy|GPU contention|GPU.*busy|gpu\d+\.lock/.test(text))return '所选显卡正在被其他任务使用。请等待空闲，或选择其他显卡后继续评测。';
  if(/inputs, code or environments changed|input identity file changed/.test(text))return '模型、代码或运行环境已改变。请新建评测，以免混用不同条件下的结果。';
  if(/output changed|plan was edited|plan changed/.test(text))return '已保存的实验计划或结果发生变化，无法继续使用。请新建评测。';
  if(/missing|not found|No such file/.test(text))return '运行所需的文件或环境不完整，请展开运行日志查看缺失项。';
  if(/lock_busy/.test(text))return '该实验仍有进程在运行，请稍后刷新。';
  if(/Failed to fetch|NetworkError|fetch failed/.test(text))return '暂时无法连接评测服务，请检查服务是否运行后重试。';
  return text;
}
function notice(message=''){
  const friendly=readableError(message);$('notice').replaceChildren(document.createTextNode(friendly));$('notice').hidden=!message;
  if(friendly!==String(message)){const details=node('details');details.append(node('summary','查看原始信息'),node('p',String(message)));$('notice').append(details);}
}
async function api(path,body,signal){
  const opts=body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json','X-Embodiq-Token':token},body:JSON.stringify(body)};
  if(signal)opts.signal=signal;
  const res=await fetch('/api/'+path,opts);const data=await res.json();
  if(!res.ok)throw new Error(data.error||`请求失败（状态码 ${res.status}）`);return data;
}
function selectedGpu(){
  const input=$('gpu');
  return !input.disabled&&input.value!==''&&input.checkValidity()?Number(input.value):null;
}
function describeGpuList(){
  const input=$('gpu'),count=[...input.options].filter(option=>option.value!=='').length;
  $('gpu-list-status').textContent=input.dataset.demo==='true'?'只读演示，显卡选项不代表本机设备。':`检测到 ${count} 张显卡，选择后查看显存。`;
}
async function refreshGpuList(){
  if(gpuListLoading)return;
  gpuListLoading=true;const input=$('gpu');
  if(input.value!=='')gpuListSelection=input.value;
  input.disabled=true;$('refresh-gpus').disabled=true;
  $('gpu-list-status').textContent='正在读取显卡列表…';
  preview();refreshGpuMemory();
  const controller=new AbortController(),timeout=setTimeout(()=>controller.abort(),4500);
  try{
    const data=await api('gpus',undefined,controller.signal);
    if(!['ok','demo'].includes(data.status))throw new Error('unavailable');
    if(!Array.isArray(data.gpus)||data.gpus.some(gpu=>!Number.isInteger(gpu)||gpu<0||gpu>255))throw new Error('invalid devices');
    const indices=[...new Set(data.gpus)].sort((a,b)=>a-b),demo=data.status==='demo';
    input.dataset.demo=String(demo);
    input.replaceChildren(...indices.map(gpu=>new Option(`显卡 ${gpu}${demo?'（演示）':''}`,String(gpu))));
    if(!indices.length){
      input.add(new Option('未检测到显卡',''));
      $('gpu-list-status').textContent=data.message||'未检测到显卡，请检查服务器设备后刷新列表。';
    }else{
      input.disabled=false;
      const missing=gpuListSelection!==null&&!indices.includes(Number(gpuListSelection));
      if(missing){input.add(new Option('请重新选择显卡',''),0);input.value='';}
      else input.value=gpuListSelection??String(indices[0]);
      if(missing)$('gpu-list-status').textContent='之前选择的显卡已不在列表中，请重新选择。';else describeGpuList();
    }
  }catch(e){
    input.replaceChildren(new Option('显卡列表暂不可用',''));input.disabled=true;
    $('gpu-list-status').textContent='暂无法读取显卡列表，请确认工作台服务已更新，或稍后刷新。';
  }finally{
    clearTimeout(timeout);gpuListLoading=false;$('refresh-gpus').disabled=false;
    await preview();refreshGpuMemory(true);
  }
}
function renderGpuMemory(data){
  const memory=data.memory,valid=data.status==='ok'&&memory&&['total_gib','used_gib','free_gib'].every(k=>Number.isFinite(memory[k]))&&memory.total_gib>0;
  $('gpu-memory-values').hidden=!valid;$('gpu-memory-meter').hidden=!valid;
  if(valid){
    for(const [id,key] of [['free','free_gib'],['used','used_gib'],['total','total_gib']])$('gpu-memory-'+id).replaceChildren(document.createTextNode(memory[key].toFixed(2)),node('small','GiB'));
    $('gpu-memory-meter').value=Math.max(0,Math.min(1,memory.used_gib/memory.total_gib));
    $('gpu-memory-meter').setAttribute('aria-valuetext',`已用 ${memory.used_gib.toFixed(2)} GiB，总计 ${memory.total_gib.toFixed(2)} GiB`);
    const date=new Date(data.sampled_at),time=Number.isNaN(date.getTime())?'':date.toLocaleTimeString('zh-CN',{hour12:false})+' 更新 · ';
    $('gpu-memory-status').textContent=time+'所选显卡 · 每 3 秒刷新';
  }else{
    for(const id of ['free','used','total'])$('gpu-memory-'+id).textContent='—';
    $('gpu-memory-meter').value=0;$('gpu-memory-meter').removeAttribute('aria-valuetext');
    $('gpu-memory-status').textContent=data.message||'暂无法读取显存，请稍后刷新。';
  }
}
async function refreshGpuMemory(force=false){
  if($('view-create').hidden||document.hidden)return;
  const input=$('gpu'),gpu=input.value;
  if(selectedGpu()===null){
    ++gpuMemoryRevision;gpuMemoryRequest?.abort();gpuMemoryRequest=null;gpuMemorySelection=null;
    renderGpuMemory({message:'选择显卡后，显示对应的显存信息。'});return;
  }
  const changed=gpuMemorySelection!==gpu;
  if(gpuMemoryRequest&&!changed&&!force)return;
  gpuMemoryRequest?.abort();const controller=new AbortController(),revision=++gpuMemoryRevision;gpuMemoryRequest=controller;gpuMemorySelection=gpu;
  if(changed)renderGpuMemory({message:'正在读取所选显卡的显存…'});
  const timeout=setTimeout(()=>controller.abort(),4500);
  try{
    const data=await api(`gpu-memory?gpu=${encodeURIComponent(String(Number(gpu)))}`,undefined,controller.signal);
    if(revision===gpuMemoryRevision&&input.value===gpu)renderGpuMemory(data);
  }catch(e){if(revision===gpuMemoryRevision)renderGpuMemory({message:'暂无法读取显存，请确认工作台服务已更新，或稍后刷新。'});}
  finally{clearTimeout(timeout);if(revision===gpuMemoryRevision)gpuMemoryRequest=null;}
}
function choices(id,values,names,enabled=()=>true,selected=()=>true){
  $(id).replaceChildren(...values.map(value=>{const label=node('label',undefined,'choice'),input=node('input'),text=node('span',names[value]||value);
    input.type='checkbox';input.value=value;input.name=id;input.disabled=!enabled(value);input.checked=enabled(value)&&selected(value);
    if(id==='suites')text.append(node('small',suiteDescriptions[value]||''));label.append(input,text);return label;}));
}
function selection(){
  const body={profile:'formal',gpu:selectedGpu(),wait_seconds:Number($('wait').value)};
  for(const name of ['models','methods','suites'])body[name]=[...$(name).querySelectorAll('input:checked')].map(x=>x.value);
  return body;
}
async function preview(){
  const revision=++previewRevision;const body=selection();$('submit').disabled=true;
  const summary=$('selection-summary');summary.replaceChildren();
  for(const [label,value] of [['模型',body.models.map(x=>modelNames[x]).join('、')],['量化方法',body.methods.map(x=>methodNames[x]).join('、')],['任务集合',body.suites.map(x=>suiteNames[x]).join('、')],['运行设备',body.gpu===null?'尚未选择':`显卡 ${body.gpu}`]])summary.append(node('dt',label),node('dd',value||'尚未选择'));
  $('profile-note').textContent='每个集合包含 10 项任务，每项重复 50 回合。';
  const missing=[['models','模型'],['methods','量化方法'],['suites','任务集合']].find(([key])=>!body[key].length);
  if(missing){$('estimate').textContent=`请至少选择一个${missing[1]}。`;return;}
  if(body.gpu===null){$('estimate').textContent='请先从列表选择显卡。';return;}
  if(!$('wait').checkValidity()){$('estimate').textContent='请填写有效的等待秒数。';return;}
  try{const data=await api('preview',body);if(revision!==previewRevision)return;
    $('estimate').textContent=`共 ${data.episodes.toLocaleString('en-US')} 回合，${data.total_stages} 个执行阶段`;$('submit').disabled=submitting;
  }catch(e){if(revision===previewRevision)$('estimate').textContent=readableError(e.message);}
}
function stageName(job){
  const [kind,model,method,suite]=job.job_id.split('.');
  if(kind==='report')return {title:'生成评测报告',context:'汇总成功率、响应时间和显存'};
  return {title:kind==='rollout'?'机器人任务评测':kind==='resource'?'速度与显存测量':'执行阶段',context:[modelNames[model]||model,methodNames[method]||method,suiteNames[suite]].filter(Boolean).join(' · ')};
}
function stageLabel(job){const name=stageName(job);return `${name.title} · ${name.context}`;}
function dateLabel(run){
  const match=run.run_id.match(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})/);
  return match?`${match[2]}-${match[3]} ${match[4]}:${match[5]}:${match[6]}`:(run.created||'时间未记录');
}
function runLabel(run){return `${dateLabel(run)} · ${run.models.map(x=>modelNames[x]||x).join(' / ')} · ${run.source==='archive'?'已完成评测':'正式评测'} · ${run.run_id.slice(-4)}`;}
function setOptions(id,options,value){
  const signature=JSON.stringify(options);if($(id).dataset.options!==signature){$(id).replaceChildren(...options.map(([text,v])=>new Option(text,v)));$(id).dataset.options=signature;}$(id).value=value;
}
function clearReport(){reportMode='performance';++reportRevision;reportData=null;reportSuite='all';$('report').replaceChildren(node('p','选择已完成的实验，点击“读取结果”查看。','empty'));}
function selectRun(id){
  if(id!==current){current=id;clearReport();$('log-stage').value='';}
  const run=allRuns.find(x=>x.run_id===id);if(run)renderStatus(run);renderRunList();
  if($('log-details').open)logs().catch(e=>notice(e.message));
}
function renderStatus(data){
  state=data;$('runs').value=current;$('report-run').value=current;
  $('phase').textContent=data.stop_requested&&data.active?'正在停止':phases[data.phase]||'状态待确认';
  $('run-meta').textContent=`${data.source==='archive'?'已完成评测':'正式评测'} · ${data.methods.map(x=>methodNames[x]||x).join('、')} · ${data.suites.map(x=>suiteNames[x]||x).join('、')}${data.gpu!==undefined?' · 显卡 '+data.gpu:''}`;
  $('run-id').textContent=data.run_id;$('count').textContent=data.source==='archive'?`共 ${data.n_episodes.toLocaleString('en-US')} 回合`:`已完成 ${data.completed} / ${data.total} 阶段`;
  $('progress').hidden=data.source==='archive';$('progress').max=data.total||1;$('progress').value=data.completed;
  $('cancel').disabled=!data.active||data.stop_requested;$('resume').disabled=data.active||data.phase==='complete';$('load-report').disabled=data.phase!=='complete'||data.active||readingReport;
  $('reasons').textContent=data.reasons.map(readableError).join('\n')||(data.phase==='interrupted'?'执行进程已退出。点击“继续评测”后，将检查已完成结果并恢复运行。':'');
  const signature=JSON.stringify(data.jobs);
  if($('stages').dataset.jobs!==signature){
    $('stages').dataset.jobs=signature;
    $('stages').replaceChildren(...data.jobs.map(job=>{const li=node('li'),name=stageName(job),left=node('div'),right=node('div',undefined,'stage-right');
      left.append(node('span',name.title,'stage-heading'),node('span',name.context,'stage-context'));
      right.append(node('span',job.attempts?`已执行 ${job.attempts} 次`:'尚未执行','stage-attempt'),node('span',phases[job.status]||'已中断',`badge ${job.status}`));
      if(job.error)left.append(node('span',readableError(job.error),'error'));li.append(left,right);return li;}));
  }
  const selected=$('log-stage').value;
  setOptions('log-stage',[['整体运行日志',''],...data.jobs.map(job=>[stageLabel(job),job.job_id])],data.jobs.some(j=>j.job_id===selected)?selected:'');
}
function renderRunList(){
  const search=$('run-search').value.trim().toLowerCase(),filter=$('run-filter').value;
  const runs=allRuns.filter(run=>{
    const text=[run.run_id,runLabel(run),...run.methods.map(x=>methodNames[x]||x),...run.suites.map(x=>suiteNames[x]||x)].join(' ').toLowerCase();
    const matches=filter==='all'||filter==='active'&&run.active||filter==='complete'&&run.phase==='complete'&&!run.active||filter==='attention'&&!run.active&&['blocked','failed','stopped','interrupted'].includes(run.phase);
    return matches&&text.includes(search);
  });
  $('run-total').textContent=allRuns.length;$('run-empty').hidden=!!runs.length;$('run-empty').textContent=allRuns.length?'没有符合筛选条件的实验。':'还没有评测记录。点击“新建评测”开始。';
  const signature=JSON.stringify([runs,current]);if($('run-rows').dataset.rows===signature)return;$('run-rows').dataset.rows=signature;
  $('run-rows').replaceChildren(...runs.map(run=>{const tr=node('tr',undefined,run.run_id===current?'selected':undefined);
    const title=node('td',run.models.map(x=>modelNames[x]||x).join(' / '));title.append(node('small',`${dateLabel(run)} · ${run.source==='archive'?'已完成评测':'正式评测'} · ${run.run_id.slice(-4)}`));
    const status=node('td');status.append(node('span',run.stop_requested&&run.active?'正在停止':phases[run.phase]||'状态待确认',`badge ${run.phase}`));
    const action=node('td'),button=node('button','查看详情','text-button');button.type='button';button.onclick=()=>{selectRun(run.run_id);$('monitor-title').scrollIntoView({block:'start'});};action.append(button);
    tr.append(title,node('td',run.methods.map(x=>methodNames[x]||x).join('、')),node('td',run.suites.map(x=>suiteNames[x]||x).join('、')),status,node('td',run.source==='archive'?`${run.n_episodes.toLocaleString('en-US')} 回合`:`${run.completed} / ${run.total} 阶段`),action);return tr;}));
}
async function logs(){
  if(!current||!$('log-details').open)return;const id=current,stage=$('log-stage').value;
  const data=await api(`runs/${encodeURIComponent(id)}/logs${stage?'?stage='+encodeURIComponent(stage):''}`);
  if(id===current&&stage===$('log-stage').value)$('logs').textContent=(data.truncated?'[日志较长，以下为最近的输出]\n':'')+(data.text||'暂无日志。');
}
async function refresh(){
  if(refreshing)return;refreshing=true;
  try{const data=await api('runs');allRuns=data.runs.filter(run=>run.profile==='formal');$('connection').textContent='已连接 · 每 3 秒更新';$('connection').className='online';
    if(!allRuns.some(run=>run.run_id===current)){current=allRuns[0]?.run_id||'';clearReport();}
    const options=allRuns.length?allRuns.map(run=>[runLabel(run),run.run_id]):[['暂无实验','']];
    setOptions('runs',options,current);setOptions('report-run',options,current);
    renderRunList();if(current){renderStatus(allRuns.find(run=>run.run_id===current));await logs();}
    if(data.errors.length)notice(`${data.errors.length} 条实验记录无法读取，请检查运行目录。`);
  }catch(e){$('connection').textContent='连接中断';$('connection').className='offline';notice(e.message);}finally{refreshing=false;}
}
function showView(){
  const requested=location.hash.slice(1),view=pageInfo[requested]?requested:'tasks';
  for(const name of Object.keys(pageInfo))$(`view-${name}`).hidden=name!==view;
  for(const link of document.querySelectorAll('nav a')){if(link.hash==='#'+view)link.setAttribute('aria-current','page');else link.removeAttribute('aria-current');}
  $('page-name').textContent=pageInfo[view][0];$('page-title').textContent=pageInfo[view][0];$('page-description').textContent=pageInfo[view][1];$('new-link').hidden=view==='create';
  document.title=`${pageInfo[view][0]} · EmbodiQ`;
  if(view==='create')refreshGpuMemory();
}
window.addEventListener('hashchange',showView);showView();
$('create-form').addEventListener('input',preview);
$('gpu').addEventListener('change',()=>{gpuListSelection=$('gpu').value||gpuListSelection;if(selectedGpu()!==null)describeGpuList();refreshGpuMemory();});
$('refresh-gpus').onclick=refreshGpuList;
$('refresh-gpu-memory').onclick=()=>refreshGpuMemory(true);
document.addEventListener('visibilitychange',()=>{if(!document.hidden)refreshGpuMemory();});
$('create-form').addEventListener('submit',async event=>{
  event.preventDefault();if(submitting)return;
  if(selectedGpu()===null||!$('create-form').reportValidity()){await preview();return;}
  submitting=true;$('submit').disabled=true;notice();
  try{const data=await api('runs',selection());current=data.run_id;clearReport();location.hash='tasks';await refresh();}
  catch(e){notice(e.message);}finally{submitting=false;await preview();}
});
$('refresh').onclick=refresh;
$('runs').onchange=()=>selectRun($('runs').value);$('report-run').onchange=()=>selectRun($('report-run').value);
$('run-search').oninput=renderRunList;$('run-filter').onchange=renderRunList;
$('log-stage').onchange=()=>logs().catch(e=>notice(e.message));$('log-details').ontoggle=()=>logs().catch(e=>notice(e.message));
for(const action of ['cancel','resume'])$(action).onclick=async()=>{
  if(action==='resume'&&(selectedGpu()===null||!$('wait').checkValidity())){notice('请在“新建评测”中选择显卡并填写有效的等待秒数，再继续评测。');return;}
  if(!current)return;$(action).disabled=true;notice();
  try{await api(`runs/${current}/${action}`,action==='resume'?{gpu:selectedGpu(),wait_seconds:Number($('wait').value)}:{});clearReport();await refresh();}
  catch(e){notice(e.message);await refresh();}
};
const finite=value=>typeof value==='number'&&Number.isFinite(value);
function metric(value,digits=2){return finite(value)?value.toFixed(digits):'—';}
function percent(value,digits=2){return finite(value)?metric(value*100,digits)+'%':'—';}
const methodShort={fp16:'原始精度',rtn:'RTN'};
// Signed change with an arrow; colour follows whether the direction is an improvement.
function deltaNode(value,unit,higherIsBetter,digits=1){
  if(!finite(value))return node('span','—','delta flat');
  const flat=Math.abs(value)<0.5*10**-digits,good=flat?null:(value>0)===higherIsBetter;
  const text=flat?`±0${unit}`:`${value>0?'▲ +':'▼ −'}${Math.abs(value).toFixed(digits)}${unit}`;
  const el=node('span',text,`delta ${flat?'flat':good?'good':'bad'}`);
  el.title=flat?'与原始精度持平':good?'优于原始精度':'劣于原始精度';return el;
}
function relative(quant,base){return finite(quant)&&finite(base)&&base!==0?100*(quant-base)/base:null;}
function table(headers,rows,numericFrom=headers.length){
  const wrap=node('div',undefined,'table-wrap'),el=node('table'),head=node('thead'),heading=node('tr'),body=node('tbody');
  headers.forEach((text,i)=>heading.append(node('th',text,i>=numericFrom?'num':undefined)));head.append(heading);
  for(const cells of rows){const tr=node('tr');cells.forEach((value,i)=>{const td=node('td',undefined,i>=numericFrom?'num':undefined);td.append(value instanceof Node?value:String(value));tr.append(td);});body.append(tr);}
  el.append(head,body);wrap.append(el);return wrap;
}
const suiteOrder=['libero_spatial','libero_object','libero_goal','libero_10'];
function svgNode(tag,attributes={},text){
  const el=document.createElementNS('http://www.w3.org/2000/svg',tag);
  for(const [key,value] of Object.entries(attributes))el.setAttribute(key,String(value));
  if(text!==undefined)el.textContent=text;return el;
}
// One floating tooltip for every chart mark carrying data-tip (pointer and keyboard).
const tip=node('div',undefined,'viz-tip');tip.hidden=true;tip.setAttribute('role','tooltip');document.body.append(tip);
function showTip(target,x,y){tip.textContent=target.dataset.tip;tip.hidden=false;const w=tip.offsetWidth,h=tip.offsetHeight;
  tip.style.left=Math.min(innerWidth-w-8,Math.max(8,x+14))+'px';tip.style.top=(y-h-12<8?y+18:y-h-12)+'px';}
document.addEventListener('pointermove',e=>{const t=e.target.closest?.('[data-tip]');if(t)showTip(t,e.clientX,e.clientY);else tip.hidden=true;});
document.addEventListener('focusin',e=>{const t=e.target.closest?.('[data-tip]');if(!t){tip.hidden=true;return;}const r=t.getBoundingClientRect();showTip(t,r.left+r.width/2,r.top);});
document.addEventListener('focusout',()=>{tip.hidden=true;});
function successDomain(cells){
  const values=cells.map(c=>c.success_rate).filter(finite);
  return [Math.max(0,Math.floor((Math.min(...values,1)*100-2)/10)*10),100];
}
// Dumbbell: hollow gray = original precision, filled blue = RTN; the connector is the change.
function successChart(title,subtitle,rows,domain,width=520){
  const section=node('section',undefined,'trend-card'),heading=node('h3',title);heading.append(node('small',subtitle));section.append(heading);
  const left=86,right=84,top=26,rowH=42,height=top+rows.length*rowH+6,[lo,hi]=domain;
  const x=v=>left+(Math.max(lo,Math.min(hi,v*100))-lo)/(hi-lo)*(width-left-right);
  const svg=svgNode('svg',{viewBox:`0 0 ${width} ${height}`,role:'group','aria-label':`${title} 成功率：原始精度与 RTN 对比`,class:'trend-svg'});
  const step=[10,20,30,50].find(t=>(width-left-right)*t/(hi-lo)>=44)||50;
  for(let t=lo;t<=hi;t+=step){const px=left+(t-lo)/(hi-lo)*(width-left-right);
    svg.append(svgNode('line',{x1:px,y1:top-6,x2:px,y2:height-4,class:t===lo?'viz-axis':'viz-grid'}),svgNode('text',{x:px,y:12,'text-anchor':'middle',class:'viz-tick'},`${t}%`));}
  rows.forEach((row,i)=>{
    const cy=top+i*rowH+rowH/2,g=svgNode('g',{class:'viz-row'}),base=row.base?.success_rate,quant=row.quant?.success_rate;
    const delta=finite(base)&&finite(quant)?(quant-base)*100:null;
    const lines=[`${title} · ${row.label}`];
    if(row.base)lines.push(`原始精度  ${percent(base,1)}（${row.base.n_success??'?'}/${row.base.n_episodes} 回合）`);
    if(row.quant)lines.push(`RTN  ${percent(quant,1)}（${row.quant.n_success??'?'}/${row.quant.n_episodes} 回合）`);
    if(finite(delta))lines.push(`成功率差值  ${delta>=0?'+':'−'}${Math.abs(delta).toFixed(2)}%（RTN − 原始精度）`);
    const hit=svgNode('rect',{x:0,y:cy-rowH/2+2,width,height:rowH-4,rx:6,class:'viz-hit',tabindex:0,role:'img','aria-label':lines.join('；')});hit.dataset.tip=lines.join('\n');
    g.append(hit,svgNode('text',{x:4,y:cy+4.5,class:'viz-row-label'},row.label));
    if(finite(base)&&finite(quant))g.append(svgNode('line',{x1:x(base),y1:cy,x2:x(quant),y2:cy,class:'viz-link'}));
    if(finite(base))g.append(svgNode('circle',{cx:x(base),cy,r:5,class:'viz-base'}));
    if(finite(quant))g.append(svgNode('circle',{cx:x(quant),cy,r:6,class:'viz-quant'}));
    const shown=finite(quant)?quant:base;
    if(finite(shown))g.append(svgNode('text',{x:width-4,y:finite(delta)?cy-2:cy+4.5,'text-anchor':'end',class:'viz-value'},percent(shown,1)));
    if(finite(delta)){const flat=Math.abs(delta)<.05;
      g.append(svgNode('text',{x:width-4,y:cy+13,'text-anchor':'end',class:flat?'viz-flat':delta>0?'viz-good':'viz-bad'},flat?'±0.0%':`${delta>0?'▲ +':'▼ −'}${Math.abs(delta).toFixed(1)}%`));}
    for(const el of g.querySelectorAll(':not(.viz-hit)'))el.setAttribute('pointer-events','none');
    svg.append(g);
  });
  section.append(svg);return section;
}
// Draw at the card's real pixel width so labels keep their font size at every viewport.
function chartWidth(){return Math.max(300,Math.min(1100,($('report').clientWidth||560)-34));}
function successCharts(cells,suite,width=chartWidth()){
  const wrap=node('div',undefined,'trend-grid'),domain=successDomain(cells);
  const find=(model,method,s)=>cells.find(c=>c.model===model&&c.method===method&&c.suite===s);
  if(suite==='all'){
    for(const model of Object.keys(modelNames)){
      const suites=suiteOrder.filter(s=>cells.some(c=>c.model===model&&c.suite===s));if(!suites.length)continue;
      wrap.append(successChart(modelNames[model],`${suites.length} 个任务集合`,suites.map(s=>({label:suiteNames[s]||s,base:find(model,'fp16',s),quant:find(model,'rtn',s)})),domain,width));
    }
  }else{
    const models=Object.keys(modelNames).filter(m=>cells.some(c=>c.model===m));
    wrap.append(successChart(suiteNames[suite]||suite,suiteDescriptions[suite]||'',models.map(m=>({label:modelNames[m],base:find(m,'fp16',suite),quant:find(m,'rtn',suite)})),domain,width));
  }
  return wrap;
}
// Headline tiles: per model, what RTN costs or saves relative to original precision.
function modelTiles(rows,successKey){
  const wrap=node('div',undefined,'result-overview');
  for(const model of Object.keys(modelNames)){
    const base=rows.find(r=>r.model===model&&r.method==='fp16'),quant=rows.find(r=>r.model===model&&r.method==='rtn');if(!base&&!quant)continue;
    const main=quant||base,tile=node('section',undefined,'model-tile'),title=node('h3',modelNames[model]||model);
    title.append(node('small',quant&&base?'RTN 相对原始精度':methodShort[main.method]));tile.append(title);
    const metrics=node('div',undefined,'tile-metrics');
    for(const [label,key,fmt,unit,higher,kind] of [['成功率',successKey,v=>metric(v*100,2),'%',true,'difference'],['峰值显存','steady_nvml_gib',v=>metric(v,1),'GiB',false,'relative'],['查询耗时','query_ms',v=>metric(v,0),'ms',false,'relative']]){
      const cell=node('div'),value=node('strong',fmt(main[key]));if(finite(main[key]))value.append(node('small',unit));cell.append(node('span',label),value);
      if(quant&&base){cell.append(kind==='difference'?deltaNode(finite(quant[key])&&finite(base[key])?(quant[key]-base[key])*100:null,'%',true,2):deltaNode(relative(quant[key],base[key]),'%',higher,0));
        cell.append(node('em',`原始 ${fmt(base[key])}${finite(base[key])?' '+unit:''}`));}
      metrics.append(cell);
    }
    tile.append(metrics);wrap.append(tile);
  }
  return wrap;
}
// Paired bars from a zero baseline; one scale per chart so models remain comparable.
function pairedBars(title,description,rows,key,format,unit){
  const section=node('section',undefined,'chart');section.append(node('h3',title),node('p',description,'hint'));
  const max=Math.max(...rows.map(r=>r[key]).filter(finite),1e-9);
  for(const model of Object.keys(modelNames)){
    const pair=['fp16','rtn'].map(m=>rows.find(r=>r.model===model&&r.method===m));if(!pair.some(Boolean))continue;
    const group=node('div',undefined,'bar-group'),head=node('div',undefined,'bar-group-head');head.append(node('span',modelNames[model]||model));
    if(pair[0]&&pair[1])head.append(deltaNode(relative(pair[1][key],pair[0][key]),'%',false,0));group.append(head);
    pair.forEach((row,i)=>{if(!row)return;const value=row[key],line=node('div',undefined,'bar-line'),track=node('div',undefined,'bar-track'),bar=node('i',undefined,`bar ${i?'quant':'base'}`);
      bar.style.width=finite(value)?`calc(${(value/max*100).toFixed(2)}% - ${(value/max*72).toFixed(1)}px)`:'0';
      track.append(bar,node('b',finite(value)?`${format(value)} ${unit}`:'—'));line.append(node('span',methodShort[row.method]||row.method),track);
      line.tabIndex=0;line.dataset.tip=`${modelNames[model]} · ${methodShort[row.method]}\n${title}  ${finite(value)?format(value)+' '+unit:'无数据'}`;group.append(line);});
    section.append(group);
  }
  return section;
}
const tradeoffMetrics={steady_nvml_gib:{label:'峰值显存',axis:'运行峰值显存（GiB）',unit:'GiB',digits:2},query_ms:{label:'查询耗时',axis:'平均查询耗时（ms）',unit:'ms',digits:1}};
const changeNote='成功率变化为两次成功率的直接差值：如 90% → 87%，显示 −3%；显存与耗时变化为相对原始精度的变化比例。';
function resourceChange(label,quant,base){
  const delta=relative(quant,base);if(!finite(delta))return null;
  if(Math.abs(delta)<1e-8)return `${label}不变`;
  const amount=Math.abs(delta)<.05?'小于 0.1%':`约 ${Math.abs(delta).toFixed(1)}%`;
  return `${label}${delta>0?'增加':'减少'}${amount}`;
}
function comparisonConclusions(rows,successKey){
  const section=node('section',undefined,'comparison-conclusions');section.setAttribute('aria-label','本次对比结论');
  section.append(node('h3','本次对比结论'),node('p','相对原始精度，按当前所选任务集合计算。','hint'));
  const list=node('ul');
  for(const model of Object.keys(modelNames)){
    const base=rows.find(r=>r.model===model&&r.method==='fp16'),quant=rows.find(r=>r.model===model&&r.method==='rtn');if(!base&&!quant)continue;
    const item=node('li'),parts=[];item.append(node('strong',modelNames[model]));
    if(!base||!quant)parts.push(`当前仅有${base?'原始精度':'RTN'}结果，缺少配对结果，暂不能计算变化`);
    else{
      if(finite(base[successKey])&&finite(quant[successKey])){
        const delta=100*(quant[successKey]-base[successKey]);
        parts.push(Math.abs(delta)<1e-8?`成功率均为 ${percent(base[successKey])}`:
          `成功率从 ${percent(base[successKey])} ${delta>0?'升至':'降至'} ${percent(quant[successKey])}（${Math.abs(delta)<.005?'差值小于 0.01%':`${delta>0?'+':'−'}${Math.abs(delta).toFixed(2)}%`}）`);
      }
      for(const [key,label] of [['steady_nvml_gib','峰值显存'],['query_ms','查询耗时']]){
        const text=resourceChange(label,quant[key],base[key]);if(text)parts.push(text);
      }
      if(!parts.length)parts.push('可对比的指标不足');
    }
    item.append(node('span',parts.join('，')+'。'));list.append(item);
  }
  section.append(list);return section;
}
function tradeoffFigure(rows,successKey,key,width=chartWidth()){
  const info=tradeoffMetrics[key],section=node('section',undefined,'tradeoff-card');
  section.append(node('h3',`成功率与${info.label}`));
  section.append(node('p','左上方表示成功率更高、资源占用更低；箭头由原始精度指向 RTN。','hint'));
  const valid=rows.filter(r=>finite(r[key])&&r[key]>=0&&finite(r[successKey])&&r[successKey]>=0&&r[successKey]<=1);
  if(!valid.length){section.append(node('p','当前选择没有同时包含成功率与资源指标的结果。','empty'));return section;}
  const height=380,left=50,right=28,top=64,bottom=60,plotWidth=width-left-right,plotHeight=height-top-bottom;
  const maximum=Math.max(...valid.map(r=>r[key]),1e-6),target=maximum*1.12/4,magnitude=10**Math.floor(Math.log10(target));
  const tickStep=([1,2,2.5,5,10].find(v=>v*magnitude>=target)||10)*magnitude,xMax=Math.ceil(maximum*1.08/tickStep)*tickStep;
  const x=value=>left+value/xMax*plotWidth,y=value=>top+(1-value)*plotHeight;
  const svg=svgNode('svg',{viewBox:`0 0 ${width} ${height}`,role:'group','aria-label':`成功率与${info.label}权衡图；横轴从零开始，纵轴为 0 至 100%`,class:'tradeoff-svg'});
  const markerId=`tradeoff-arrow-${key}`,defs=svgNode('defs'),marker=svgNode('marker',{id:markerId,viewBox:'0 0 8 8',refX:7,refY:4,markerWidth:7,markerHeight:7,orient:'auto',markerUnits:'userSpaceOnUse'});
  marker.append(svgNode('path',{d:'M1 1 L7 4 L1 7',class:'tradeoff-arrowhead'}));defs.append(marker);svg.append(defs);
  svg.append(svgNode('text',{x:left,y:18,class:'viz-row-label'},'任务成功率（%）'));
  for(const value of [0,.25,.5,.75,1])svg.append(svgNode('line',{x1:left,x2:width-right,y1:y(value),y2:y(value),class:value===0?'viz-axis':'viz-grid'}),svgNode('text',{x:left-10,y:y(value)+4,'text-anchor':'end',class:'viz-tick'},`${value*100}%`));
  for(let i=0;i<=Math.round(xMax/tickStep);i++){
    const value=i*tickStep,px=x(value),label=Number(value.toPrecision(3)).toLocaleString('en-US',{maximumSignificantDigits:3});
    svg.append(svgNode('line',{x1:px,x2:px,y1:top,y2:height-bottom,class:i===0?'viz-axis':'viz-grid'}),svgNode('text',{x:px,y:height-bottom+22,'text-anchor':'middle',class:'viz-tick'},label));
  }
  svg.append(svgNode('text',{x:left+plotWidth/2,y:height-9,'text-anchor':'middle',class:'viz-row-label'},info.axis));
  const occupied=[];
  for(const model of Object.keys(modelNames)){
    const pair=['fp16','rtn'].map(method=>valid.find(r=>r.model===model&&r.method===method)),[base,quant]=pair;
    if(!base&&!quant)continue;
    if(base&&quant){
      const dx=x(quant[key])-x(base[key]),dy=y(quant[successKey])-y(base[successKey]),length=Math.hypot(dx,dy);
      if(length>22)svg.append(svgNode('line',{x1:x(base[key])+dx/length*8,y1:y(base[successKey])+dy/length*8,x2:x(quant[key])-dx/length*10,y2:y(quant[successKey])-dy/length*10,class:'tradeoff-arrow','marker-end':`url(#${markerId})`,'data-model':model}));
    }
    for(const row of pair.filter(Boolean)){
      const px=x(row[key]),py=y(row[successKey]),label=`${modelNames[model]} · ${methodShort[row.method]}\n成功率 ${percent(row[successKey])}\n${info.label} ${metric(row[key],info.digits)} ${info.unit}`;
      const group=svgNode('g',{class:'tradeoff-point',tabindex:0,role:'img','aria-label':label.replaceAll('\n','；'),'data-tip':label,'data-model':model,'data-method':row.method,'data-resource':row[key],'data-success':row[successKey]});
      group.append(svgNode('circle',{cx:px,cy:py,r:14,class:'tradeoff-hit'}),svgNode('circle',{cx:px,cy:py,r:row.method==='fp16'?7:5,class:row.method==='fp16'?'viz-base':'viz-quant'}));svg.append(group);
    }
    const points=pair.filter(Boolean),cx=points.reduce((sum,r)=>sum+x(r[key]),0)/points.length,cy=Math.min(...points.map(r=>y(r[successKey])));
    const labelWidth=modelNames[model].length*7+10,lx=Math.max(left,Math.min(width-right-labelWidth,cx-labelWidth/2));
    let ly=cy-16;
    while(occupied.some(b=>lx<b.x+b.width&&lx+labelWidth>b.x&&Math.abs(ly-b.y)<18))ly+=20;
    occupied.push({x:lx,y:ly,width:labelWidth});svg.append(svgNode('text',{x:lx,y:ly,class:'tradeoff-model'},modelNames[model]));
  }
  section.append(svg);
  if(valid.length<rows.length)section.append(node('p','缺少成功率或资源指标的结果未绘入图中。','hint'));
  return section;
}
function tradeoffPanel(rows,successKey){
  const section=node('section',undefined,'tradeoff-panel'),controls=node('div',undefined,'tradeoff-controls'),buttons=node('div',undefined,'chart-mode');buttons.setAttribute('aria-label','权衡图横轴');
  controls.append(node('span','资源指标','muted'));
  for(const [key,info] of Object.entries(tradeoffMetrics)){
    const button=node('button',info.label);button.type='button';button.setAttribute('aria-pressed',String(tradeoffMetric===key));button.onclick=()=>{tradeoffMetric=key;renderReport();$('report').querySelector('.tradeoff-controls button[aria-pressed="true"]')?.focus();};buttons.append(button);
  }
  controls.append(buttons);section.append(controls,tradeoffFigure(rows,successKey,tradeoffMetric));
  if(tradeoffMetric==='query_ms')section.append(node('p','查询耗时用于同一模型量化前后比较；不同模型一次输出的动作数量不同，不能据此直接判断控制速度。','hint'));
  const download=node('button','下载当前权衡图（SVG）','text-button');download.type='button';download.disabled=!section.querySelector('svg');download.onclick=()=>downloadTradeoffChart(rows,successKey);section.append(download);
  return section;
}
function legend(rows){
  const el=node('div',undefined,'chart-legend');
  for(const [method,cls,label] of [['fp16','base','原始精度（BF16）'],['rtn','quant','RTN 四位量化']])if(rows.some(r=>r.method===method)){const item=node('span');item.append(node('i',undefined,`key ${cls}`),document.createTextNode(label));el.append(item);}
  return el;
}
// Standalone SVG export: inline the light-theme tokens so the file renders anywhere.
const exportStyle='.viz-grid{stroke:#eceef1}.viz-axis{stroke:#c9cdd4}.viz-tick{fill:#6b7280;font-size:11px}.viz-row-label{fill:#454b56;font-size:13px}.viz-value{fill:#14171c;font-size:13px;font-weight:600}.viz-link{stroke:#d3d7de;stroke-width:2;stroke-linecap:round}.viz-base{fill:#fff;stroke:#8d929b;stroke-width:2}.viz-quant{fill:#2a78d6;stroke:#fff;stroke-width:2}.viz-good{fill:#127a3e;font-size:11px;font-weight:600}.viz-bad{fill:#c0362f;font-size:11px;font-weight:600}.viz-flat{fill:#6b7280;font-size:11px;font-weight:600}.viz-hit{fill:none}text{font-family:system-ui,-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif}';
function downloadSuccessChart(cells,suite){
  const cards=[...successCharts(cells,suite,560).children],width=600;let y=96;
  const doc=svgNode('svg',{xmlns:'http://www.w3.org/2000/svg',width});doc.append(svgNode('style',{},exportStyle),svgNode('rect',{width:'100%',height:'100%',fill:'#fff'}));
  doc.append(svgNode('text',{x:32,y:40,fill:'#14171c','font-size':20,'font-weight':600},'EmbodiQ · 任务成功率：原始精度 vs RTN'));
  doc.append(svgNode('circle',{cx:38,cy:66,r:5,class:'viz-base'}),svgNode('text',{x:50,y:70,class:'viz-tick'},'原始精度（BF16）'),svgNode('circle',{cx:170,cy:66,r:6,class:'viz-quant'}),svgNode('text',{x:182,y:70,class:'viz-tick'},'RTN 四位量化 · 右侧为 RTN 成功率及变化'));
  for(const card of cards){
    const svg=card.querySelector('svg'),[, , w,h]=svg.getAttribute('viewBox').split(' ').map(Number);
    doc.append(svgNode('text',{x:32,y:y+18,fill:'#14171c','font-size':15,'font-weight':600},card.querySelector('h3').firstChild.textContent));
    svg.setAttribute('x',32);svg.setAttribute('y',y+28);svg.setAttribute('width',width-64);svg.setAttribute('height',h*(width-64)/w);doc.append(svg);
    y+=28+h*(width-64)/w+24;
  }
  doc.append(svgNode('text',{x:32,y:y+8,class:'viz-tick'},'成功率变化 = RTN − 原始精度；例如 90% → 87%，显示 −3%。'));y+=32;
  doc.setAttribute('height',y);doc.setAttribute('viewBox',`0 0 ${width} ${y}`);
  const url=URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(doc)],{type:'image/svg+xml'}));
  const link=node('a');link.href=url;link.download='success-rates.svg';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
function downloadTradeoffChart(rows,successKey){
  const width=900,chart=tradeoffFigure(rows,successKey,tradeoffMetric,width-64).querySelector('svg');if(!chart)return;
  const height=530,doc=svgNode('svg',{xmlns:'http://www.w3.org/2000/svg',width,height,viewBox:`0 0 ${width} ${height}`});
  doc.append(svgNode('style',{},exportStyle+'.tradeoff-arrow{stroke:#8d929b;stroke-width:1.6}.tradeoff-arrowhead{fill:none;stroke:#8d929b;stroke-width:1.4}.tradeoff-hit{fill:transparent}.tradeoff-model{fill:#454b56;font-size:12px;font-weight:600}'),svgNode('rect',{width:'100%',height:'100%',fill:'#fff'}));
  doc.append(svgNode('text',{x:32,y:34,fill:'#14171c','font-size':20,'font-weight':600},`EmbodiQ · 成功率与${tradeoffMetrics[tradeoffMetric].label}`));
  doc.append(svgNode('text',{x:32,y:57,class:'viz-tick'},reportSuite==='all'?'当前所选任务集合均值':suiteNames[reportSuite]));
  doc.append(svgNode('circle',{cx:40,cy:80,r:6,class:'viz-base'}),svgNode('text',{x:54,y:84,class:'viz-tick'},'原始精度（BF16）'),svgNode('circle',{cx:196,cy:80,r:5,class:'viz-quant'}),svgNode('text',{x:210,y:84,class:'viz-tick'},'RTN 四位量化 · 箭头由原始精度指向 RTN'));
  chart.setAttribute('x',32);chart.setAttribute('y',100);chart.setAttribute('width',width-64);chart.setAttribute('height',380);doc.append(chart);
  doc.append(svgNode('text',{x:32,y:508,class:'viz-tick'},tradeoffMetric==='query_ms'?'查询耗时用于同一模型量化前后比较；不同模型一次输出的动作数量不同。':'左上方表示成功率更高、资源占用更低。缺失指标不绘入图中。'));
  const url=URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(doc)],{type:'image/svg+xml'}));
  const link=node('a');link.href=url;link.download=`tradeoff-${tradeoffMetric==='query_ms'?'latency':'memory'}.svg`;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
function renderReport(){
  if(!reportData)return;const data=reportData,container=$('report');container.replaceChildren();tip.hidden=true;reportWidth=chartWidth();
  const order=(list,key)=>value=>list.indexOf(value[key]);
  const cells=data.cells.filter(c=>reportSuite==='all'||c.suite===reportSuite).sort((a,b)=>order(Object.keys(modelNames),'model')(a)-order(Object.keys(modelNames),'model')(b)||order(suiteOrder,'suite')(a)-order(suiteOrder,'suite')(b)||order(['fp16','rtn'],'method')(a)-order(['fp16','rtn'],'method')(b));
  const rows=reportSuite==='all'?data.aggregates:cells,successKey=reportSuite==='all'?'selected_suites_success_rate':'success_rate';
  const label=node('div',undefined,`report-label${data.formal?' formal':''}`);
  label.append(node('span',data.formal?'✓ 正式评测 · 结果已通过完整性检查':'! 结果未满足正式评测条件'));
  label.append(node('span',`${new Set(data.cells.map(c=>c.model+c.method)).size} 个模型/方法组合 · ${new Set(data.cells.map(c=>c.suite)).size} 个任务集合 · ${data.cells.reduce((sum,c)=>sum+c.n_episodes,0).toLocaleString('en-US')} 回合`,'meta'));
  container.append(label);
  const toolbar=node('div',undefined,'report-toolbar'),tabs=node('div',undefined,'suite-tabs');tabs.setAttribute('aria-label','筛选任务集合');
  for(const [value,text] of [['all','全部集合'],...suiteOrder.filter(s=>data.cells.some(c=>c.suite===s)).map(s=>[s,suiteNames[s]||s])]){const button=node('button',text);button.type='button';button.setAttribute('aria-pressed',String(reportSuite===value));button.onclick=()=>{reportSuite=value;renderReport();[...$('report').querySelectorAll('.suite-tabs button')].find(b=>b.getAttribute('aria-pressed')==='true')?.focus();};tabs.append(button);}
  toolbar.append(tabs,node('span',reportSuite==='all'?'成功率与耗时为所选集合均值，显存为峰值':'仅显示当前集合','muted'));container.append(toolbar);
  container.append(modelTiles(rows,successKey));
  container.append(node('p',changeNote,'change-note'));
  const modes=node('div',undefined,'chart-heading'),buttons=node('div',undefined,'chart-mode');buttons.setAttribute('aria-label','图表类型');
  for(const [value,text] of [['performance','任务成功率'],['resources','速度与显存'],['tradeoffs','性能与资源权衡']]){
    const button=node('button',text);button.type='button';button.setAttribute('aria-pressed',String(reportMode===value));button.onclick=()=>{reportMode=value;renderReport();[...$('report').querySelectorAll('.chart-mode button')].find(b=>b.getAttribute('aria-pressed')==='true')?.focus();};buttons.append(button);
  }
  modes.append(buttons,legend(rows));container.append(modes);
  if(reportMode==='performance'){
    reportWidth=chartWidth();
    const [lo]=successDomain(cells);
    container.append(node('p',`每行连线表示量化前后的变化；横轴从 ${lo}% 起，以放大差异。右侧为 RTN 成功率及与原始精度的差值。`,'chart-description'));
    container.append(successCharts(cells,reportSuite));
  }else if(reportMode==='tradeoffs'){
    container.append(tradeoffPanel(rows,successKey));
  }else{
    container.append(node('p','条形从零开始；组标题旁为 RTN 相对原始精度的变化，越低越好。','chart-description'));
    const charts=node('div',undefined,'chart-grid');
    charts.append(pairedBars('运行峰值显存','GiB · 越低越节省显存',rows,'steady_nvml_gib',v=>metric(v,2),'GiB'),pairedBars('平均查询耗时','毫秒 / 次 · 越短越好；适合在同一模型内比较',rows,'query_ms',v=>metric(v,1),'ms'));
    container.append(charts);
  }
  container.append(comparisonConclusions(rows,successKey));
  if(reportSuite==='all'){
    container.append(node('h3','模型与方法对比','report-section'));
    container.append(table(['模型','量化方法','所选集合平均成功率','四集合平均成功率','成功率差值','平均查询耗时（ms）','运行峰值显存（GiB）'],data.aggregates.map(r=>[modelNames[r.model]||r.model,methodNames[r.method]||r.method,percent(r.selected_suites_success_rate),percent(r.four_suite_macro_success_rate),r.method==='fp16'?'基准':deltaNode(r.delta_pp_vs_fp16,'%',true,2),metric(r.query_ms,1),metric(r.steady_nvml_gib)]),2));
    container.append(node('p','“—”表示当前实验不满足统计条件，或缺少原始精度对照；不表示得分为零。','hint'));
  }
  container.append(node('h3','分集合结果','report-section'));
  container.append(table(['模型','量化方法','任务集合','成功率','回合数','平均响应（ms）','运行峰值显存（GiB）','加载峰值显存（GiB）'],cells.map(c=>[modelNames[c.model]||c.model,methodNames[c.method]||c.method,suiteNames[c.suite]||c.suite,percent(c.success_rate),c.n_episodes.toLocaleString('en-US'),metric(c.query_ms),metric(c.steady_nvml_gib,3),metric(c.load_alloc_gib,3)]),3));
  container.append(node('p','运行峰值来自显卡监测，加载峰值来自推理框架，二者统计口径不同。各指标说明见下方“指标怎么看？”。','hint'));
  const links=node('div',undefined,'downloads');for(const [file,text] of [['report.md','下载完整报告'],['cells.csv','下载数据表'],['tradeoffs.svg','下载原始对比图']]){const a=node('a',text);a.href=`/api/runs/${current}/report/${file}`;a.download=file;links.append(a);}container.append(links);
  const chartDownload=node('button','下载成功率对比图（SVG）','text-button');chartDownload.type='button';chartDownload.onclick=()=>downloadSuccessChart(cells,reportSuite);links.append(chartDownload);
  const details=node('details',undefined,'technical');details.append(node('summary','数据说明'));
  for(const caveat of data.caveats||[])details.append(node('p',caveat));container.append(details);
}
let reportWidth=0;
window.addEventListener('resize',()=>{clearTimeout(window.reportResize);window.reportResize=setTimeout(()=>{if(reportData&&reportMode!=='resources'&&Math.abs(chartWidth()-reportWidth)>8){reportWidth=chartWidth();renderReport();}},150);});
$('load-report').onclick=async()=>{
  if(!current||readingReport)return;const id=current,revision=++reportRevision;readingReport=true;$('load-report').disabled=true;$('load-report').textContent='正在读取…';notice();
  try{const data=await api(`runs/${id}/report`);if(current!==id||revision!==reportRevision)return;reportData=data;reportSuite='all';renderReport();}
  catch(e){notice(e.message);}finally{readingReport=false;$('load-report').textContent='读取结果';if(state)$('load-report').disabled=state.phase!=='complete'||state.active;}
};
(async()=>{
  try{const caps=await api('capabilities');token=caps.csrf_token;
    choices('models',caps.models,modelNames);choices('methods',Object.keys(caps.methods),methodNames,x=>caps.methods[x]==='ready');choices('suites',caps.suites,suiteNames);
    await refreshGpuList();await refresh();setInterval(refresh,3000);setInterval(()=>refreshGpuMemory(),3000);
  }catch(e){notice('页面初始化失败：'+e.message);}
})();
