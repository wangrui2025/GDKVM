import { guide } from "../../lib/agentReader";
export const prerender=true;
export function GET(){return new Response(guide,{headers:{"Content-Type":"text/markdown; charset=utf-8"}});}
