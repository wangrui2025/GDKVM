import { llms } from "../lib/agentReader";
export const prerender=true;
export function GET(){return new Response(llms,{headers:{"Content-Type":"text/plain; charset=utf-8"}});}
