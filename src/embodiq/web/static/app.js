'use strict';
const $ = id => document.getElementById(id);
const modelNames = {openvla:'OpenVLA', vla_adapter:'VLA-Adapter', fastwam:'FastWAM'};
const methodNames = {fp16:'原始精度', rtn:'四位量化（RTN）', awq:'AWQ · 待接入', hsa:'HSA · 待接入'};
const suiteNames = {libero_spatial:'空间关系', libero_object:'目标物体', libero_goal:'任务目标', libero_10:'长时程操作'};
const suiteDescriptions = {libero_spatial:'理解物体的位置关系',libero_object:'识别并操作指定物体',libero_goal:'根据指令完成目标',libero_10:'连续完成多步操作'};
const phases = {planned:'等待开始',starting:'正在启动',preflight:'检查模型与输入',waiting_for_gpu:'等待空闲显卡',running:'正在评测',complete:'已完成',blocked:'需要处理',failed:'运行失败',stopped:'已停止',interrupted:'运行中断',pending:'等待开始'};
const pageInfo = {create:['新建评测','选择模型、量化方法和任务集合。'],tasks:['任务管理','查看评测进度，管理已提交的实验。'],results:['评测结果','按任务集合比较成功率、响应时间和显存占用。']};
let token='', current='', state=null, allRuns=[], refreshing=false, previewRevision=0, reportRevision=0, submitting=false, reportData=null, reportSuite='all', readingReport=false;
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
async function api(path,body){
  const opts=body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json','X-Embodiq-Token':token},body:JSON.stringify(body)};
  const res=await fetch('/api/'+path,opts);const data=await res.json();
  if(!res.ok)throw new Error(data.error||`请求失败（状态码 ${res.status}）`);return data;
}
function choices(id,values,names,enabled=()=>true,selected=()=>true){
  $(id).replaceChildren(...values.map(value=>{const label=node('label',undefined,'choice'),input=node('input'),text=node('span',names[value]||value);
    input.type='checkbox';input.value=value;input.name=id;input.disabled=!enabled(value);input.checked=enabled(value)&&selected(value);
    if(id==='suites')text.append(node('small',suiteDescriptions[value]||''));label.append(input,text);return label;}));
}
function selection(){
  const body={profile:'formal',gpu:Number($('gpu').value),wait_seconds:Number($('wait').value)};
  for(const name of ['models','methods','suites'])body[name]=[...$(name).querySelectorAll('input:checked')].map(x=>x.value);
  return body;
}
async function preview(){
  const revision=++previewRevision;const body=selection();$('submit').disabled=true;
  const summary=$('selection-summary');summary.replaceChildren();
  for(const [label,value] of [['模型',body.models.map(x=>modelNames[x]).join('、')],['量化方法',body.methods.map(x=>methodNames[x]).join('、')],['任务集合',body.suites.map(x=>suiteNames[x]).join('、')],['运行设备',`显卡 ${body.gpu}`]])summary.append(node('dt',label),node('dd',value||'尚未选择'));
  $('profile-note').textContent='每个集合包含 10 项任务，每项重复 50 回合。';
  const missing=[['models','模型'],['methods','量化方法'],['suites','任务集合']].find(([key])=>!body[key].length);
  if(missing){$('estimate').textContent=`请至少选择一个${missing[1]}。`;return;}
  if(!$('gpu').checkValidity()||!$('wait').checkValidity()){$('estimate').textContent='请填写有效的显卡编号和等待秒数。';return;}
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
function clearReport(){++reportRevision;reportData=null;reportSuite='all';$('report').replaceChildren(node('p','选择已完成的实验，点击“读取结果”查看。','empty'));}
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
  try{const data=await api('runs');allRuns=data.runs.filter(run=>run.profile==='formal');$('connection').textContent='已连接 · 每 3 秒更新';
    if(!allRuns.some(run=>run.run_id===current)){current=allRuns[0]?.run_id||'';clearReport();}
    const options=allRuns.length?allRuns.map(run=>[runLabel(run),run.run_id]):[['暂无实验','']];
    setOptions('runs',options,current);setOptions('report-run',options,current);
    renderRunList();if(current){renderStatus(allRuns.find(run=>run.run_id===current));await logs();}
    if(data.errors.length)notice(`${data.errors.length} 条实验记录无法读取，请检查运行目录。`);
  }catch(e){$('connection').textContent='连接中断';notice(e.message);}finally{refreshing=false;}
}
function showView(){
  const requested=location.hash.slice(1),view=pageInfo[requested]?requested:'tasks';
  for(const name of Object.keys(pageInfo))$(`view-${name}`).hidden=name!==view;
  for(const link of document.querySelectorAll('nav a')){if(link.hash==='#'+view)link.setAttribute('aria-current','page');else link.removeAttribute('aria-current');}
  $('page-name').textContent=pageInfo[view][0];$('page-title').textContent=pageInfo[view][0];$('page-description').textContent=pageInfo[view][1];$('new-link').hidden=view==='create';
  document.title=`${pageInfo[view][0]} · EmbodiQ`;
}
window.addEventListener('hashchange',showView);showView();
$('create-form').addEventListener('input',preview);
$('create-form').addEventListener('submit',async event=>{
  event.preventDefault();if(submitting)return;submitting=true;$('submit').disabled=true;notice();
  try{const data=await api('runs',selection());current=data.run_id;clearReport();location.hash='tasks';await refresh();}
  catch(e){notice(e.message);}finally{submitting=false;await preview();}
});
$('refresh').onclick=refresh;
$('runs').onchange=()=>selectRun($('runs').value);$('report-run').onchange=()=>selectRun($('report-run').value);
$('run-search').oninput=renderRunList;$('run-filter').onchange=renderRunList;
$('log-stage').onchange=()=>logs().catch(e=>notice(e.message));$('log-details').ontoggle=()=>logs().catch(e=>notice(e.message));
for(const action of ['cancel','resume'])$(action).onclick=async()=>{
  if(!current)return;$(action).disabled=true;notice();
  try{await api(`runs/${current}/${action}`,action==='resume'?{gpu:Number($('gpu').value),wait_seconds:Number($('wait').value)}:{});clearReport();await refresh();}
  catch(e){notice(e.message);await refresh();}
};
function metric(value,digits=2){return typeof value==='number'&&Number.isFinite(value)?value.toFixed(digits):'—';}
function percent(value){return typeof value==='number'&&Number.isFinite(value)?metric(value*100)+'%':'—';}
function table(headers,rows){
  const wrap=node('div',undefined,'table-wrap'),el=node('table'),head=node('thead'),heading=node('tr'),body=node('tbody');
  headers.forEach(text=>heading.append(node('th',text)));head.append(heading);
  for(const cells of rows){const tr=node('tr');cells.forEach(text=>tr.append(node('td',String(text),'numeric')));body.append(tr);}
  el.append(head,body);wrap.append(el);return wrap;
}
function chart(title,description,rows,key,format,fixedMax){
  const section=node('section',undefined,'chart');section.append(node('h3',title),node('p',description,'hint'));
  const values=rows.map(row=>row[key]).filter(value=>typeof value==='number'&&Number.isFinite(value));const max=fixedMax||Math.max(...values,0.001);
  for(const row of rows){const line=node('div',undefined,'chart-row'),caption=node('div',undefined,'chart-caption');
    caption.append(node('span',`${modelNames[row.model]||row.model} · ${row.method==='fp16'?'原始精度':methodNames[row.method]||row.method}`),node('strong',format(row[key])));line.append(caption);
    const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.setAttribute('viewBox','0 0 100 6');svg.setAttribute('preserveAspectRatio','none');svg.setAttribute('aria-hidden','true');svg.classList.add('chart-track');
    for(const [width,cls] of [[100,'chart-bg'],[typeof row[key]==='number'&&Number.isFinite(row[key])?Math.max(0,Math.min(100,row[key]/max*100)):0,`chart-bar${row.method==='fp16'?' baseline':''}`]]){const rect=document.createElementNS(svg.namespaceURI,'rect');rect.setAttribute('width',String(width));rect.setAttribute('height','6');rect.setAttribute('rx','1');rect.setAttribute('class',cls);svg.append(rect);}
    line.append(svg);section.append(line);
  }return section;
}
function renderReport(){
  if(!reportData)return;const data=reportData,container=$('report');container.replaceChildren();
  container.append(node('p',data.display_note||(data.formal?'正式评测 · 保存的结果已通过完整性检查':'结果未满足正式评测条件'),`report-label${data.formal?' formal':''}`));
  const toolbar=node('div',undefined,'report-toolbar'),tabs=node('div',undefined,'suite-tabs');tabs.setAttribute('aria-label','筛选任务集合');
  const suites=[...new Set(data.cells.map(c=>c.suite))];
  for(const [value,label] of [['all','全部集合'],...suites.map(s=>[s,suiteNames[s]||s])]){const button=node('button',label);button.type='button';button.setAttribute('aria-pressed',String(reportSuite===value));button.onclick=()=>{reportSuite=value;renderReport();[...$('report').querySelectorAll('.suite-tabs button')].find(b=>b.getAttribute('aria-pressed')==='true')?.focus();};tabs.append(button);}
  toolbar.append(tabs,node('span',reportSuite==='all'?'汇总当前实验所选集合':'仅显示当前集合','muted'));container.append(toolbar);
  const cells=data.cells.filter(c=>reportSuite==='all'||c.suite===reportSuite);
  const rows=reportSuite==='all'?data.aggregates.map(r=>({...r,success_rate:r.selected_suites_success_rate})):cells;
  const charts=node('div',undefined,'chart-grid');charts.append(chart('任务成功率',reportSuite==='all'?'所选集合的平均值 · 越高越好':'当前集合 · 越高越好',rows,'success_rate',percent,1),chart('平均响应时间','每次模型请求 · 毫秒 · 越短越好',rows,'query_ms',v=>metric(v)),chart('运行峰值显存','显卡监测 · GiB · 越低越好',rows,'steady_nvml_gib',v=>metric(v,3)));container.append(charts);
  if(reportSuite==='all'){
    container.append(node('h3','模型与方法对比','report-section'));
    container.append(table(['模型','量化方法','所选集合平均成功率','四集合平均成功率','较原始精度变化（百分点）'],data.aggregates.map(r=>[modelNames[r.model]||r.model,methodNames[r.method]||r.method,percent(r.selected_suites_success_rate),percent(r.four_suite_macro_success_rate),metric(r.delta_pp_vs_fp16)])));
    container.append(node('p','“—”表示当前实验不满足统计条件，或缺少原始精度对照；不表示得分为零。','hint'));
  }
  container.append(node('h3','分集合结果','report-section'));
  container.append(table(['模型','量化方法','任务集合','成功率','回合数','平均响应（毫秒）','运行峰值显存（GiB）','加载峰值显存（GiB）'],cells.map(c=>[modelNames[c.model]||c.model,methodNames[c.method]||c.method,suiteNames[c.suite]||c.suite,percent(c.success_rate),c.n_episodes,metric(c.query_ms),metric(c.steady_nvml_gib,3),metric(c.load_alloc_gib,3)])));
  container.append(node('p','运行峰值来自显卡监测，加载峰值来自推理框架，二者统计口径不同。各指标说明见下方“指标怎么看？”。','hint'));
  const links=node('div',undefined,'downloads');for(const [file,label] of [['report.md','下载完整报告'],['cells.csv','下载数据表'],['tradeoffs.svg','下载原始对比图']]){const a=node('a',label);a.href=`/api/runs/${current}/report/${file}`;a.download=file;links.append(a);}container.append(links);
  const details=node('details',undefined,'technical');details.append(node('summary','数据说明'));
  for(const caveat of data.caveats||[])details.append(node('p',caveat));container.append(details);
}
$('load-report').onclick=async()=>{
  if(!current||readingReport)return;const id=current,revision=++reportRevision;readingReport=true;$('load-report').disabled=true;$('load-report').textContent='正在读取…';notice();
  try{const data=await api(`runs/${id}/report`);if(current!==id||revision!==reportRevision)return;reportData=data;reportSuite='all';renderReport();}
  catch(e){notice(e.message);}finally{readingReport=false;$('load-report').textContent='读取结果';if(state)$('load-report').disabled=state.phase!=='complete'||state.active;}
};
(async()=>{
  try{const caps=await api('capabilities');token=caps.csrf_token;
    choices('models',caps.models,modelNames);choices('methods',Object.keys(caps.methods),methodNames,x=>caps.methods[x]==='ready');choices('suites',caps.suites,suiteNames);
    await preview();await refresh();setInterval(refresh,3000);
  }catch(e){notice('页面初始化失败：'+e.message);}
})();
