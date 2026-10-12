/* Guided navigation over verified historical results; never submits a model job. */
(() => {
  'use strict';
  const source=JSON.parse(JSON.stringify(window.EMBODIQ_DEMO));
  const original=Object.values(source.reports)[0];
  const runId=source.runs[0].run_id;
  function subset(request){
    const report=JSON.parse(JSON.stringify(original));
    report.cells=report.cells.filter(c=>request.models.includes(c.model)&&request.methods.includes(c.method)&&request.suites.includes(c.suite));
    report.n_episodes=report.cells.reduce((sum,c)=>sum+c.n_episodes,0);
    report.aggregates=[];
    for(const model of request.models)for(const method of request.methods){
      const cells=report.cells.filter(c=>c.model===model&&c.method===method);
      if(!cells.length)continue;
      const average=key=>cells.reduce((sum,c)=>sum+c[key],0)/cells.length;
      const baseline=report.cells.filter(c=>c.model===model&&c.method==='fp16');
      const success=average('success_rate');
      report.aggregates.push({model,method,selected_suites_success_rate:success,
        four_suite_macro_success_rate:cells.length===4?success:null,
        delta_pp_vs_fp16:baseline.length?100*(success-baseline.reduce((sum,c)=>sum+c.success_rate,0)/baseline.length):null,
        query_ms:average('query_ms'),per_action_ms:average('per_action_ms'),
        steady_nvml_gib:Math.max(...cells.map(c=>c.steady_nvml_gib))});
    }
    report.display_note=`已完成评测的配置示例 · ${report.cells.length} 个组合 · ${report.n_episodes.toLocaleString('en-US')} 回合`;
    return report;
  }
  window.embodiqDemoSubset=subset;
  const steps=document.createElement('div');steps.className='demo-steps';steps.setAttribute('aria-label','演示步骤');
  steps.innerHTML='<a href="#create"><span>1</span><div>配置评测<small>选择模型、方法与任务</small></div></a><a href="#tasks"><span>2</span><div>查看任务<small>核对配置与评测范围</small></div></a><a href="#results"><span>3</span><div>对比结果<small>查看指标并导出报告</small></div></a>';
  document.getElementById('notice').before(steps);
  document.getElementById('submit').textContent='查看本配置示例';
  document.querySelector('.after-submit').textContent='按本页选择筛选已完成结果，不启动模型。';
  document.querySelector('#view-tasks .actions').hidden=true;
  document.getElementById('log-details').hidden=true;
  document.getElementById('stages').hidden=true;
  const coverage=document.createElement('section');coverage.className='demo-coverage';coverage.id='demo-coverage';
  document.getElementById('stages').after(coverage);
  const next=document.createElement('div');next.className='demo-next';
  next.innerHTML='<p>配置核对完成后，查看这些任务的成功率、耗时和显存。</p><a class="button primary" href="#results">下一步：对比结果 →</a>';
  coverage.after(next);
  function showCoverage(){
    const report=window.EMBODIQ_DEMO.reports[runId],run=window.EMBODIQ_DEMO.runs[0];
    coverage.replaceChildren();
    const title=document.createElement('h3');title.textContent='本配置已完成的评测内容';coverage.append(title);
    const grid=document.createElement('div');grid.className='demo-coverage-grid';
    for(const [label,value,note] of [
      ['任务评测',`${report.cells.length} 个组合`,`${report.n_episodes.toLocaleString('en-US')} 回合 · 每个组合 10 项任务，每项 50 回合`],
      ['资源测量',`${run.models.length*run.methods.length} 个模型 / 方法配置`,'使用固定输入，分别记录查询耗时和峰值显存'],
      ['评测报告','结果已齐全','按任务集合对比，可下载报告和数据表']]){
      const card=document.createElement('div');
      for(const [tag,text] of [['span',label],['strong',value],['p',note]]){const el=document.createElement(tag);el.textContent=text;card.append(el);}
      grid.append(card);
    }
    coverage.append(grid);
    const note=document.createElement('p');note.className='hint';note.textContent='此处展示已有实验数据。实际运行时，任务管理会显示执行进度，并支持停止、续跑和查看日志。';coverage.append(note);
  }
  document.getElementById('create-form').addEventListener('submit',async event=>{
    event.preventDefault();event.stopImmediatePropagation();
    const request=selection();
    if(['models','methods','suites'].some(k=>!request[k].length))return;
    const report=subset(request);
    window.EMBODIQ_DEMO.reports[runId]=report;
    Object.assign(window.EMBODIQ_DEMO.runs[0],{models:request.models,methods:request.methods,suites:request.suites,n_episodes:report.n_episodes});
    clearReport();await refresh();showCoverage();location.hash='tasks';window.scrollTo(0,0);
  },true);
  async function showStep(){
    const view=location.hash.slice(1)||'create';
    for(const link of steps.querySelectorAll('a')){
      if(link.hash==='#'+view)link.setAttribute('aria-current','step');else link.removeAttribute('aria-current');
    }
    if(view==='results'&&current&&!reportData&&!document.getElementById('load-report').disabled)document.getElementById('load-report').click();
  }
  window.addEventListener('hashchange',()=>{showStep();window.scrollTo(0,0);});
  new MutationObserver(()=>showStep()).observe(document.getElementById('load-report'),{attributes:true,attributeFilter:['disabled']});
  showCoverage();showStep();
})();
