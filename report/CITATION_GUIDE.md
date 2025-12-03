# Citation Guide for R Markdown

## Overview
Your report uses **BibTeX** for citations, which is the standard for academic R Markdown documents. I've already set everything up for you!

## How It Works

### 1. The `.bib` File (references.bib)
This file contains all your references in BibTeX format. I've added:

- **@Cont2014** - The original paper you're replicating
- **@GitHubCopilot2025** - AI assistance acknowledgment

### 2. Citing in Your R Markdown Document

Use these formats to cite:

#### Standard Citation (Author Year)
```markdown
Cont, Kukanov, and Stoikov [-@Cont2014] show that...
```
**Renders as:** Cont, Kukanov, and Stoikov (2014) show that...

#### Parenthetical Citation
```markdown
OFI predicts price changes [@Cont2014].
```
**Renders as:** OFI predicts price changes (Cont, Kukanov & Stoikov, 2014).

#### Multiple Citations
```markdown
Previous studies [@Cont2014; @GitHubCopilot2025] have shown...
```

#### Suppress Author Name (Year only)
```markdown
Cont et al. [-@Cont2014] found...
```
**Renders as:** Cont et al. (2014) found...

## What I've Already Done

✅ **Added to `references.bib`:**
```bibtex
@Article{Cont2014,
  author    = {Cont, Rama and Kukanov, Arseniy and Stoikov, Sasha},
  title     = {The Price Impact of Order Book Events},
  journal   = {Journal of Financial Econometrics},
  year      = {2014},
  ...
}

@Misc{GitHubCopilot2025,
  author       = {{GitHub Inc.}},
  title        = {GitHub Copilot (Large Language Model)},
  year         = {2025},
  ...
}
```

✅ **Updated your R Markdown document:**
- Introduction cites `[-@Cont2014]` (line ~27)
- Paper Summary cites `[-@Cont2014]` (line ~48)
- Added Acknowledgments section before References (line ~780)
- Acknowledgments cite both `[@GitHubCopilot2025]` and `[@Cont2014]`

✅ **YAML header already configured:**
```yaml
bibliography: "references.bib"
```

## Where Citations Appear

When you cite `[@Cont2014]` anywhere in your document, the reference will:
1. Appear as (Cont, Kukanov & Stoikov, 2014) in text
2. Automatically appear in the References section at the end
3. Be formatted according to the journal style

## Rendering Your Document

To generate the final PDF with citations:

### Option 1: RStudio (Recommended)
1. Open `ReplicationProjectTemplate.Rmd` in RStudio
2. Click **"Knit"** button at the top
3. Citations will be automatically processed

### Option 2: Command Line (if you have R installed)
```powershell
cd "d:\Harshu\UIUC\Education\Semester 3\FIN 554-Algo Trading Sys Design & Testing\Assignments\ofi-replication\report"
Rscript -e "rmarkdown::render('ReplicationProjectTemplate.Rmd')"
```

### Option 3: Pandoc (for advanced users)
```powershell
pandoc ReplicationProjectTemplate.Rmd --citeproc --bibliography=references.bib -o output.pdf
```

## Common Citation Patterns in Your Report

### Throughout the document:
- "Cont et al. [-@Cont2014] demonstrate..." 
- "...as shown by @Cont2014"
- "Previous research [@Cont2014] indicates..."
- "AI assistance [@GitHubCopilot2025] was used for..."

## APA Guidelines for AI Citations

The acknowledgment I added follows emerging standards for AI disclosure:
- **Transparency:** Clearly state what AI was used for
- **Specificity:** List specific tasks (debugging, testing, documentation)
- **Responsibility:** Affirm that analysis and conclusions are your own
- **Citation:** Provide formal reference with access information

Many universities now require AI disclosure. Your acknowledgment section handles this professionally.

## Troubleshooting

### "Citation not found" error?
- Check that the citation key matches exactly (case-sensitive): `@Cont2014` not `@cont2014`

### Reference not appearing?
- Ensure you've cited it somewhere in the text with `[@key]`
- Or add to `nocite` in YAML header (already done for some references)

### Want to add more references?
Add them to `references.bib` following the BibTeX format:
```bibtex
@Article{AuthorYear,
  author  = {Last, First and Last2, First2},
  title   = {Title of Paper},
  journal = {Journal Name},
  year    = {2024},
  volume  = {10},
  pages   = {1--20},
  doi     = {10.xxxx/xxxxx},
}
```

## Summary

✅ Everything is already set up!
✅ Just use `[@Cont2014]` to cite the paper
✅ AI assistance is acknowledged in the Acknowledgments section
✅ All references will appear automatically when you knit/render

No need to worry about formatting—BibTeX handles it all!
