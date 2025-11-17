import json, csv, os, sys

csv_path = sys.argv[1]  # columns: wav_path,answer_text
out_path = sys.argv[2]
system_prompt = ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
                 "capable of perceiving auditory and visual inputs, as well as generating text and speech.")
items=[]
with open(csv_path) as f:
    r=csv.DictReader(f)
    for row in r:
        wav=row["wav_path"]
        ans=row.get("answer_text","")
        items.append({
          "conversations":[
            {"from":"system","value":system_prompt},
            {"from":"human","value":"Describe the spatial sound events you hear: <audio>"},
            {"from":"gpt","value":ans}
          ],
          "audios":[wav]
        })
with open(out_path,"w") as f: json.dump(items,f,ensure_ascii=False,indent=2)
print(f"Wrote {len(items)} items to {out_path}")
