# Fit the final STM (prevalence ~ venue * s(year)) and extract:
#  - theta, topic labels (prob + FREX words), representative docs
#  - venue-specific prevalence trajectories with 95% CIs (method of composition)
#  - OOPSLA difference-in-differences around the 2007 CFP change and 2010 SPLASH rebrand
suppressMessages({library(stm); library(data.table)})
args <- commandArgs(TRUE); K <- as.integer(args[1])
inp <- readRDS("out/stm_input_core4.rds"); meta <- inp$meta
set.seed(2015)
if (file.exists(sprintf("out/stm_K%d.rds", K))) fit <- readRDS(sprintf("out/stm_K%d.rds", K)) else fit <- stm(inp$documents, inp$vocab, K = K, prevalence = ~ venue * s(year, df = 5), data = meta,
           init.type = "Spectral", max.em.its = 300, verbose = FALSE)
if (!file.exists(sprintf("out/stm_K%d.rds", K))) saveRDS(fit, sprintf("out/stm_K%d.rds", K))
th <- as.data.table(fit$theta); setnames(th, sprintf("T%02d", 1:K)); th[, doc_id := inp$doc_id]
fwrite(th, sprintf("out/stm_K%d_theta.csv", K))
lab <- labelTopics(fit, n = 12)
fwrite(data.table(topic = 1:K, prob = apply(lab$prob, 1, paste, collapse = ", "),
                  frex = apply(lab$frex, 1, paste, collapse = ", "),
                  lift = apply(lab$lift, 1, paste, collapse = ", "),
                  prevalence = colMeans(fit$theta)), sprintf("out/stm_K%d_labels.csv", K))
fwrite(as.data.table(exp(fit$beta$logbeta[[1]])), sprintf("out/stm_K%d_beta.csv", K))
fwrite(data.table(word = inp$vocab), "out/stm_vocab.csv")
tc <- topicCorr(fit, method = "simple", cutoff = 0.02); fwrite(as.data.table(tc$cor), sprintf("out/stm_K%d_topiccorr.csv", K))

# venue x year trajectories
ee <- estimateEffect(1:K ~ venue * s(year, df = 5), fit, metadata = meta, uncertainty = "Global", nsims = 50)
years <- 1973:2026; out <- list()
for (v in levels(meta$venue)) {
  yr <- range(meta$year[meta$venue == v]); ys <- yr[1]:yr[2]
  nd <- data.frame(venue = factor(v, levels = levels(meta$venue)), year = ys)
  X <- as.matrix(stm:::makeDesignMatrix(~ venue * s(year, df = 5), meta, nd))
  for (k in 1:K) {
    # simulate from the global approximate posterior of the regression coefficients
    draws <- do.call(cbind, lapply(ee$parameters[[k]], function(p) X %*% t(MASS::mvrnorm(4, p$est, p$vcov))))
    out[[length(out) + 1]] <- data.table(venue = v, year = ys, topic = k, mean = rowMeans(draws),
                                         lo = apply(draws, 1, quantile, .025), hi = apply(draws, 1, quantile, .975))
  }
}
fwrite(rbindlist(out), sprintf("out/stm_K%d_trajectories.csv", K))

# OOPSLA DiD window 1997-2019, year fixed effects absorb field-wide trends
w <- meta$year >= 1997 & meta$year <= 2019
md <- data.frame(oopsla = as.numeric(meta$venue == "oopsla"), post07 = as.numeric(meta$year >= 2007),
                 post10 = as.numeric(meta$year >= 2010), fy = factor(meta$year), year = meta$year)[w, ]
fitw <- fit; fitw$theta <- fit$theta[w, ]
did <- estimateEffect(1:K ~ oopsla * post07 + oopsla * post10 + fy, fitw, metadata = md, uncertainty = "None")
s <- summary(did); rows <- list()
for (k in 1:K) { tb <- s$tables[[k]]; for (term in c("oopsla:post07", "oopsla:post10"))
  rows[[length(rows) + 1]] <- data.table(topic = k, term = term, est = tb[term, 1], se = tb[term, 2], p = tb[term, 4]) }
fwrite(rbindlist(rows), sprintf("out/stm_K%d_oopsla_did.csv", K))
cat("done\n")
