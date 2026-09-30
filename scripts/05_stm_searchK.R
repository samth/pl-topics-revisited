# Structural Topic Model: choose K by held-out likelihood, residuals, semantic coherence, exclusivity.
suppressMessages({library(stm); library(data.table)})
d <- fread("data/corpus_tokens.csv")[core4 == TRUE]
toks <- strsplit(d$tokens, " ", fixed = TRUE)
vocab <- sort(unique(unlist(toks)))
documents <- lapply(toks, function(t) { tb <- table(match(t, vocab)); rbind(as.integer(names(tb)), as.integer(tb)) })
meta <- data.frame(venue = factor(d$venue), year = d$year)
saveRDS(list(documents = documents, vocab = vocab, meta = meta, doc_id = d$doc_id), "out/stm_input_core4.rds")
set.seed(42)
sk <- searchK(documents, vocab, K = c(10, 15, 20, 25, 30, 40, 50, 60), prevalence = ~ venue * s(year, df = 5),
              data = meta, init.type = "Spectral", max.em.its = 150, verbose = FALSE, cores = 2)
r <- as.data.frame(lapply(sk$results, unlist))
print(r); fwrite(r, "out/stm_searchK.csv")
