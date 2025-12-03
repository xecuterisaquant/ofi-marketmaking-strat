# Render R Markdown report to PDF
library(rmarkdown)

# Render the document
render("OFI-MarketMaker-Report.Rmd")

cat("\nPDF generation complete!\n")
