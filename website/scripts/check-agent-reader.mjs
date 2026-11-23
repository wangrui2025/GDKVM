import assert from "node:assert/strict";
import fs from "node:fs";
const read=p=>fs.readFileSync("dist/"+p,"utf8");
const r=JSON.parse(read("agent/site.json"));
const llms=read("llms.txt"),skill=read("agent/SKILL.md");
assert.equal(r.schema,"academic-project/agent-reader/v1");
assert.equal(r.site,"GDKVM");
assert.deepEqual(Object.keys(r).sort(),["schema","site","access","publication","pages","readerPages","discovery","cautions"].sort());
const base="/GDKVM/";
for(const l of ["en","zh"]){
  const h=read(l+"/index.html"),a=read(l+"/agent/index.html");
  assert.ok(h.includes(base+l+"/agent/"));
  assert.ok(h.includes(base+"agent/SKILL.md"));
  assert.ok(a.includes("data-agent-context"));
  assert.ok(a.includes(base+"agent/site.json"));
}
for(const p of r.pages){
  assert.ok(p.path.startsWith(base));
  assert.ok(llms.includes(p.path));
  assert.ok(fs.existsSync("dist/"+p.path.slice(base.length)+"index.html"));
}
for(const p of r.publication.sources){const u=new URL(p.url);assert.equal(u.protocol,"https:");assert.ok(!u.username&&!u.password);assert.ok(llms.includes(p.url));}
assert.match(skill,/^---\nname: /);
assert.doesNotMatch(JSON.stringify(r)+llms+skill,/\/Users\/|\/home\/|mykcs\/\.agents|[A-Z_]+_TOKEN\s*=/);
assert.match(read("agent/index.html"),/noindex/);
const data=JSON.parse(fs.readFileSync("src/content/homepage/en.json","utf8"));
assert.equal(r.publication.title.en,data.publicationTitle);
assert.equal(r.publication.venue.en,data.venueFull);
console.log("PASS public Agent directory, source links, locale navigation and static routes");
