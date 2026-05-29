# StatsPAI Arellano-Bond GMM parity (R side) -- Module 50.
#
# Reads data/50_xtabond.csv and runs plm::pgmm with one-step
# first-difference GMM. This is intentionally a secondary reference:
# Stata's xtabond remains the byte-level migration fixture for Module 50
# because plm uses a different finite-sample vcov stack. The point
# estimates should nevertheless stay in the same specification family.

.args <- commandArgs(trailingOnly = FALSE)
.file_arg <- grep("^--file=", .args, value = TRUE)
.script_dir <- if (length(.file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", .file_arg[1])))
} else {
  getwd()
}
source(file.path(.script_dir, "_common.R"))

suppressPackageStartupMessages({
  library(plm)
})

MODULE <- "50_xtabond"

df <- read_csv_strict(MODULE)
pdf <- plm::pdata.frame(df, index = c("id", "time"))

# Formula mirrors sp.xtabond(..., lags=1, gmm_lags=(2, None)):
#   D.y_it = rho D.y_{i,t-1} + beta D.x_it + D.e_it
# instrumented with available lagged levels L(2/.).y.
fit <- plm::pgmm(
  y ~ lag(y, 1) + x | lag(y, 2:99),
  data = pdf,
  effect = "individual",
  model = "onestep",
  transformation = "d"
)

smry <- summary(fit, robust = TRUE)
co <- coef(smry)

rows <- list(
  parity_row(
    MODULE,
    "beta_y_lag",
    estimate = unname(co["lag(y, 1)", "Estimate"]),
    se = unname(co["lag(y, 1)", "Std. Error"]),
    n = nrow(df)
  ),
  parity_row(
    MODULE,
    "beta_x",
    estimate = unname(co["x", "Estimate"]),
    se = unname(co["x", "Std. Error"]),
    n = nrow(df)
  )
)

write_results(MODULE, rows,
              extra = list(method = "Arellano-Bond difference GMM",
                           step = "one-step",
                           vcov = "robust",
                           package = "plm::pgmm",
                           note = paste(
                             "Secondary R reference; Stata xtabond remains",
                             "the strict Module 50 reference because plm",
                             "uses a different finite-sample vcov stack."
                           )))
