import en from "../content/homepage/en.json";
import zh from "../content/homepage/zh.json";
import { SITE_CONFIG } from "../config/site";

export const base = import.meta.env.BASE_URL.replace(/\/$/, "");
export const href = (path: string) => base + "/" + path.replace(/^\//, "");
const routes = [
  {
    "path": "",
    "en": "Paper homepage",
    "zh": "论文主页"
  },
  {
    "path": "tool/",
    "en": "Visualization tool",
    "zh": "可视化工具"
  },
  {
    "path": "reprod/",
    "en": "Reproduction guide",
    "zh": "复现指南"
  }
] as const;
export const reader = {
  schema: "academic-project/agent-reader/v1",
  site: "GDKVM",
  access: "public, read-only",
  publication: {
    title: {en:en.publicationTitle,zh:zh.publicationTitle},
    venue: {en:en.venueFull,zh:zh.venueFull},
    sources: [{title:"arXiv",url:SITE_CONFIG.arxivUrl},{title:"CVF",url:SITE_CONFIG.cvfPaperUrl},{title:"Public code",url:SITE_CONFIG.codeUrl}]
  },
  pages: routes.flatMap(r => (["en","zh"] as const).map(lang=>({lang,title:r[lang],path:href(lang+"/"+r.path)}))),
  readerPages:{en:href("en/agent/"),zh:href("zh/agent/")},
  discovery:{index:href("llms.txt"),catalog:href("agent/site.json"),guide:href("agent/SKILL.md")},
  cautions:{
    en:["Read the paper and primary evidence before stating scientific conclusions. A directory or visualization is not a replication result or clinical validation.","Do not infer hidden datasets, patient data or private experiments. Reading grants no permission to execute code, contact anyone or publish."],
    zh:["先读论文及原始证据，再评价研究结论。目录、可视化或演示不等于独立复现或临床验证。","不推测未公开数据、患者资料或私人实验。阅读不授予运行代码、联系他人或发布的权限。"]
  }
};
export const llms = [
  "# "+reader.site,"","> Public bilingual academic project navigation; the paper is the authority for reported results.","",
  "## Reading resources",...Object.entries(reader.discovery).filter(([k])=>k!=="index").map(([k,p])=>"- ["+k+"]("+p+")"),
  ...Object.entries(reader.readerPages).map(([l,p])=>"- [Reader "+l+"]("+p+")"),"",
  "## Project pages",...reader.pages.map(r=>"- ["+r.title+"]("+r.path+")"),"",
  "## Primary sources",...reader.publication.sources.map(x=>"- ["+x.title+"]("+x.url+")"),"",
  "Resolve site paths against the origin from which this file was retrieved.",...reader.cautions.en,...reader.cautions.zh,""
].join("\n");
export const guide = [
  "---","name: gdkvm-reader","description: Read the public bilingual GDKVM site and cite original evidence.","---","",
  "# Agent reading guide / 阅读指南","",
  "1. Start at "+reader.discovery.index+" and "+reader.discovery.catalog+".",
  "2. Choose the language and page appropriate to the question.",
  "3. Open and cite the primary paper; distinguish illustrations from measured results.","",
  "先选语言并打开对应页面；引用时给出论文或具体来源链接，不用摘要替代原始证据。",
  ...reader.cautions.en,...reader.cautions.zh,""
].join("\n");
export function copyContext(origin:string){
  return ["# "+reader.site+" — public project context",origin+base+"/","",...reader.pages.map(x=>"- "+x.title+": "+origin+x.path),"",...reader.publication.sources.map(x=>"- "+x.title+": "+x.url),"",...reader.cautions.en,...reader.cautions.zh].join("\n");
}
