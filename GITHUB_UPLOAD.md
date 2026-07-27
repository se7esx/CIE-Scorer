# Upload to GitHub

Run these commands from the parent directory that contains `CIE-scorer`:

```bash
cd CIE-scorer
git init
git branch -M main
git add .
git commit -m "Initial release of CIE-Scorer"
git remote add origin https://github.com/se7esx/CIE-Scorer.git
git push -u origin main
```

If the remote already contains files, pull them first:

```bash
git pull origin main --allow-unrelated-histories
git push -u origin main
```

If you prefer SSH:

```bash
git remote set-url origin git@github.com:se7esx/CIE-Scorer.git
git push -u origin main
```

After installation, the FaithCoT-BENCH entry point is:

```bash
cie-score-faithcot --help
```
