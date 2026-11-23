import { reader } from "../../lib/agentReader";
export const prerender=true;
export function GET(){return new Response(JSON.stringify(reader,null,2)+"\n",{headers:{"Content-Type":"application/json; charset=utf-8"}});}
