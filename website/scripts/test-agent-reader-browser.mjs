import assert from "node:assert/strict";
import net from "node:net";
import { spawn } from "node:child_process";
import { setTimeout as sleep } from "node:timers/promises";
import { chromium } from "playwright";
import fs from "node:fs";
const catalog=JSON.parse(fs.readFileSync("dist/agent/site.json","utf8"));
const base=catalog.discovery.index.replace(/\/llms\.txt$/,"");
const listener=net.createServer();
await new Promise((resolve,reject)=>listener.once("error",reject).listen(0,"127.0.0.1",resolve));
const port=listener.address().port;
await new Promise(resolve=>listener.close(resolve));
const origin="http://127.0.0.1:"+port;
const child=spawn(process.execPath,["node_modules/astro/bin/astro.mjs","preview","--host","127.0.0.1","--port",String(port)],{stdio:"pipe"});
let log="";
child.stdout.on("data",x=>log+=x.toString());
child.stderr.on("data",x=>log+=x.toString());
let browser;
try{
  let ready=false;
  for(let i=0;i<100;i++){
    if(child.exitCode!==null)throw Error(log.slice(-600));
    try{if((await fetch(origin+catalog.discovery.catalog)).ok){ready=true;break}}catch{}
    await sleep(150);
  }
  assert.ok(ready,"preview not reachable");
  browser=await chromium.launch({channel:"chrome",headless:true});
  for(const lang of ["en","zh"]) for(const [name,viewport,dark] of [
    ["desktop",{width:1440,height:900},false],
    ["phone",{width:440,height:956},false],
    ["dark",{width:1440,height:900},true]
  ]){
    const ctx=await browser.newContext({viewport,deviceScaleFactor:name==="phone"?3:1,isMobile:name==="phone",hasTouch:name==="phone",colorScheme:dark?"dark":"light",permissions:["clipboard-read","clipboard-write"]});
    const page=await ctx.newPage();
    const errors=[];
    page.on("pageerror",e=>errors.push(e.message));
    const res=await page.goto(origin+base+"/"+lang+"/",{waitUntil:"domcontentloaded"});
    assert.equal(res.status(),200);
    await page.locator('footer a[href="'+catalog.readerPages[lang]+'"]').click();
    await page.waitForURL(origin+catalog.readerPages[lang]);
    await page.locator("[data-agent-reader]").waitFor();
    const text=await page.locator("[data-agent-context]").inputValue();
    assert.ok(text.includes(catalog.publication.sources[0].url));
    await page.locator("[data-agent-copy]").click();
    await page.waitForFunction(()=>document.querySelector("[data-agent-status]")?.textContent?.length>0);
    assert.equal(await page.evaluate(()=>navigator.clipboard.readText()),text);
    const metrics=await page.evaluate(()=>({width:innerWidth,scroll:document.documentElement.scrollWidth}));
    assert.ok(metrics.scroll<=metrics.width+1,"horizontal overflow");
    await page.goto(origin+catalog.readerPages[lang]);
    await page.evaluate(()=>Object.defineProperty(navigator.clipboard,"writeText",{configurable:true,value:async()=>{throw Error("clipboard denied in test")}}));
    await page.locator("[data-agent-copy]").click();
    await page.waitForFunction(()=>document.querySelector("[data-agent-details]")?.open);
    assert.ok(await page.locator("[data-agent-context]").evaluate(x=>x.selectionEnd>x.selectionStart));
    assert.deepEqual(errors,[]);
    await ctx.close();
    console.log("PASS "+catalog.site+" "+lang+" "+name+" homepage navigation / real copy / manual fallback / no overflow");
  }
  for(const lang of ["en","zh"]){
    const ctx=await browser.newContext({javaScriptEnabled:false,viewport:{width:440,height:956}});
    const page=await ctx.newPage();
    await page.goto(origin+catalog.readerPages[lang]);
    await page.locator("[data-agent-details] summary").click();
    assert.ok((await page.locator("[data-agent-context]").inputValue()).includes(catalog.publication.sources[0].url));
    await ctx.close();
    console.log("PASS "+catalog.site+" "+lang+" no-JS copy");
  }
}finally{await browser?.close();child.kill("SIGTERM");}
