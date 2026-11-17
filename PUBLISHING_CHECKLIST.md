# GitHub publishing checklist

1. Create a repository named `nursing-workforce-forecasting-ts` (private at first if you wish to review it).
2. Upload only the contents of this prepared package, not any internal workforce datasets, contracts, private development materials or credentials.
3. Confirm that the public dataset has just two columns: `DATANOM` and `PPE`, and is explicitly labeled synthetic.
4. Re-run the three notebooks in order. Check that no local information appears in outputs. The included output cells were run only on synthetic data.
5. Verify model comparisons and exported prediction tables match `outputs/reports/`.
6. Decide licensing separately; no open-source license is assigned by default.
7. Publish the GitHub URL and replace the pending-link placeholder in the Spanish blog draft.

## Commands after creating an empty repository

```bash
git init
git add .
git status
git commit -m "Add synthetic nursing workforce SARIMA tutorial"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/nursing-workforce-forecasting-ts.git
git push -u origin main
```

Do not commit any serialized local model; model pickle files are excluded by `.gitignore` and should be generated only on your computer.
