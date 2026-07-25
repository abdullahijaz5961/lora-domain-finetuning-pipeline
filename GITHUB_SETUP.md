# GitHub setup

Recommended repository name: `lora-domain-finetuning-pipeline`

```powershell
git init
git branch -M main
git add .
git commit -m "feat: launch LoRA Domain Fine-Tuning Pipeline"
git remote add origin https://github.com/abdullahijaz5961/lora-domain-finetuning-pipeline.git
git push -u origin main
```

After changing a file directly on GitHub, run `git pull origin main` before the next local push.
Never commit `.env`, credentials, customer data, private documents, or large model weights.
