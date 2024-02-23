library(rjson)
library(reticulate)
torch <- import("torch")

##############################################################
### Constants and functions
##############################################################

BASEDIR <- sprintf('%s/github/llm-thinking-english/src/bob', Sys.getenv('HOME'))
PLOTDIR <- sprintf('%s/plots/', BASEDIR)
SIZES <- c('7b', '13b', '70b')
LANGS <- c('de', 'fr', 'ru')
COLS <- list(en='#ff7f0e', de='#1f77b4')

# Set this to FALSE if you don't want to save plots to PDFs.
SAVE_PLOTS <- FALSE

# Before multiplying with U, need to RMS-normalize:
# https://github.com/huggingface/transformers/blob/f9f1f2ac5e03bba46d18cfc5df30472b2b85ba54/src/transformers/models/llama/modeling_llama.py#L112
rmsnorm <- function(mtx) {
  eps <- 1e-6
  std <- sqrt(colMeans(mtx^2) + eps)
  t((t(mtx) / std))
}

# Sanity check to see whether rmsnorm works (this test is for the 70b model):
test_rmsnorm <- function() {
  i <- as.vector(read.table(sprintf('%s/data/latents/probe_input.csv', BASEDIR))[,1])
  o <- as.vector(read.table(sprintf('%s/data/latents/probe_output.csv', BASEDIR))[,1])
  plot(t(matrices$unemb[['70b']]) %*% rmsnorm(matrix(i, ncol=1, nrow=8192)), o)
  abline(0,1)
}

# This one has latents only for all tokens in the sequence.
load_latents__OLD <- function(size, pair) {
  file <- sprintf('%s/data/latents/block_output_latents_%s_%s.pt', BASEDIR, size, pair)
  model <- torch$load(file)
  
  # Load the weights (thanks to ChatGPT...).
  lat <- as.array(model$data$float()$numpy())
  lat <- lat[,,,]
  
  lapply(1:dim(lat)[1], function(i) lat[i,,])
}

# This one has latents only for the last token in the sequence.
load_latents <- function(size, pair) {
  file <- sprintf('%s/data/latents/%s_%s_latents.pt', BASEDIR, size, pair)
  model <- torch$load(file)
  
  # Load the weights (thanks to ChatGPT...).
  lat <- as.array(model$data$float()$numpy())
  lat <- lat[,,]
  
  lapply(1:dim(lat)[1], function(i) t(lat[i,,]))
}

# Turn token embeddings (mtx1) and latents (mtx2) into probability distributions
# (one distribution per column). With dims1, we can select a subset of the
# tokens from the distribution (after normalization), i.e., a subset of rows from
# the output matrix.
compute_probs <- function(mtx1, mtx2, dims1) {
  # Compute logits.
  Logits <- t(mtx1) %*% mtx2
  # Compute unnormalized probs.
  Exp <- exp(Logits)
  # Compute partition function.
  Norm <- colSums(Exp)
  # Normalize.
  Prob <- t(t(Exp) / Norm)
  # Return the required rows.
  Prob[dims1,]
}

##############################################################
### Load data
##############################################################

tok_ids <- sort(sapply(fromJSON(file=sprintf('%s/data/tokens.json', BASEDIR)),
                       function(x) x))
toks <- names(tok_ids)
names(toks) <- tok_ids

inputs <- list()
for (lang in LANGS) {
  i <- read.csv(sprintf('%s/data/inputs/clean/%s_%s.csv', BASEDIR, lang, lang), header=FALSE,
                col.names=c('out_token_id','out_token_str','latent_token_id',
                            'latent_token_str','in_token_id','in_token_str'),
                colClasses=c('integer', 'character'))
  i$tok <- toks[i$out_token_id+1]
  i$tok_en <- toks[i$latent_token_id+1]
  inputs[[lang]] <- i
}

# sigma is the vector with which the latent is multiplied pointwise before multiplying
# with the unembedding mtx U. This is equivalent to scaling the rows (features) of U,
# so below we do this and then forget about sigma.
sigmas <- list()
for (size in SIZES) {
  sigmas[[size]] <- as.vector(read.table(sprintf('%s/data/latents/rmsnorm_%s.csv', BASEDIR, size))[,1])
}

file <- sprintf('%s/data/RData/emb_unemb_matrices.RData', BASEDIR)
if (!file.exists(file)) {
  matrices <- list()
  for (type in c('emb', 'unemb')) {
    matrices[[type]] <- list()
    for (size in SIZES) {
      cat(sprintf('%s %s\n', type, size))
      X <- read.table(pipe(sprintf('gunzip -c %s/data/emb_unemb_matrices/%s_%s.txt.gz',
                                   BASEDIR, type, size)), sep=',')
      rownames(X) <- toks
      X <- t(X)
      matrices[[type]][[size]] <- X
    }
  }
  save(matrices, file=file)
} else {
  load(file)
  # Multiply (pointwise) the features of U with sigma.
  for (size in SIZES) {
    matrices[['unemb']][[size]] <- sigmas[[size]] * matrices[['unemb']][[size]]
  }
}


##############################################################
### Sanity check: reproduce main plots from paper (for de_de repeat task)
##############################################################

size <- '13b'
U <- matrices$unemb[[size]]
latents <- load_latents(size, 'de_de')
nlay <- ncol(latents[[1]])
L <- rmsnorm(do.call(cbind, latents))

ProbUL <- compute_probs(U, L, 1:ncol(U))

# Turns a token into its variants, in particular w/ and w/o leading "▁".
# If expand==FALSE, does nothing, but simply returns the input.
get_token_variant_ids <- function(tok, expand=FALSE) {
  if (!expand) {
    variants <- tok_ids[tok] + 1
  } else {
    without_underscore <- sub('^▁', '', tok)
    with_underscore <- sprintf('▁%s', without_underscore)
    variants <- c()
    if (without_underscore %in% toks) variants <- c(variants, tok_ids[without_underscore] + 1)
    if (with_underscore %in% toks) variants <- c(variants, tok_ids[with_underscore] + 1)
  }
  variants
}

# curves <- lapply(1:length(latents), function(i) {
#   cols <- ((i-1)*nlay+1):(i*nlay)
#   rows <- c(tok_de[i], tok_en[i])
#   ProbUL[rows, cols]
# })

curves <- lapply(1:length(latents), function(i) {
  cols <- ((i-1)*nlay+1):(i*nlay)
  expand <- TRUE
  rows_de <- get_token_variant_ids(inputs$de$tok[i], expand=expand)
  rows_en <- get_token_variant_ids(inputs$de$tok_en[i], expand=expand)
  prob_de <- matrix(ProbUL[rows_de, cols], ncol=nlay)
  prob_en <- matrix(ProbUL[rows_en, cols], ncol=nlay)
  rbind(colSums(prob_de), colSums(prob_en))
})

avg_de <- colMeans(do.call(rbind, lapply(curves, function(m) m[1,])))
avg_en <- colMeans(do.call(rbind, lapply(curves, function(m) m[2,])))

# With lines for individual paths.
matplot(1:nlay-1, do.call(cbind, lapply(curves, function(m) m[1,])), type='l',
        col=rgb(0,0,1,0.08), lty=1, xlab='Layer', ylab='Probability', bty='n',
        panel.first=abline(v=seq(0,40,5), col='gray', lty=3))
matlines(1:nlay-1, do.call(cbind, lapply(curves, function(m) m[2,])), type='l',
         col=rgb(1,0.65,0,0.08), lty=1)
lines(1:nlay-1, avg_de, col='#1f77b4', lwd=3)
lines(1:nlay-1, avg_en, col='#ff7f0e', lwd=3)
legend('topleft', legend=c('de', 'en'), col=c('#1f77b4', '#ff7f0e'), lty=1, bty='n')

# Only averages.
if (SAVE_PLOTS) pdf(sprintf('%s/lang_probs.pdf', PLOTDIR), width=3.6, height=1.6,
                    pointsize=12, family='Helvetica')
par(mar=c(3.2, 3.5, 0.8, 0.7))
plot(1:nlay-1, avg_de, col='#1f77b4', lwd=3, type='l', bty='n', xlab='', ylab='',
     axes=FALSE, ylim=c(0,1))
lines(1:nlay-1, avg_en, col='#ff7f0e', lwd=3)
axis(1, at=seq(0,nlay,5))
mtext('Layer', side=1, line=2)
axis(2, las=1)
mtext('Lang. prob.', side=2, line=2.4)
legend('topleft', legend=c('en', 'de'), col=c('#ff7f0e', '#1f77b4'), lty=1, lwd=3,
       bty='n')
if (SAVE_PLOTS) dev.off()


### Entropy

ent <- apply(ProbUL, 2, function(p) -sum(p*log2(p)))
ent <- matrix(ent, nrow=nlay, ncol=length(latents))

if (SAVE_PLOTS) pdf(sprintf('%s/entropy.pdf', PLOTDIR), width=3.6, height=1.6,
                    pointsize=12, family='Helvetica')
par(mar=c(3.2, 3.5, 0.8, 0.7))
plot(1:nlay-1, rowMeans(ent), col='black', lwd=3, type='l', bty='n', xlab='', ylab='',
     axes=FALSE, ylim=c(0,15))
axis(1, at=seq(0,nlay,5))
mtext('Layer', side=1, line=2)
axis(2, at=seq(0,15,5), las=1)
mtext('Entropy (bits)', side=2, line=2.2)
if (SAVE_PLOTS) dev.off()


##############################################################
### Make plots for sphere diagram
##############################################################

# Using data from Chris, for translate-to-ZH task.

size <- '70b'
x <- read.csv(sprintf('%s/data/data_for_sphere_plots/%s_zh_lines.csv', BASEDIR, size))
nlay <- nrow(x)

# Lang probs.
if (SAVE_PLOTS) pdf(sprintf('%s/lang_probs_%s.pdf', PLOTDIR, size), width=3.6, height=1.6,
                    pointsize=12, family='Helvetica')
par(mar=c(3.2, 3.5, 0.8, 0.7))
plot(1:nlay, x$zh, col='#1f77b4', lwd=3, type='l', bty='n', xlab='', ylab='',
     axes=FALSE, ylim=range(0,0.8))
lines(1:nlay, x$en, col='#ff7f0e', lwd=3)
axis(1, at=seq(0,nlay,20))
mtext('Layer', side=1, line=2)
axis(2, las=1)
mtext('Lang. prob.', side=2, line=2.4)
legend('topleft', legend=c('en', 'zh'), col=c('#ff7f0e', '#1f77b4'), lty=1, lwd=3,
       bty='n')
if (SAVE_PLOTS) dev.off()

# Entropy.
if (SAVE_PLOTS) pdf(sprintf('%s/entropy_%s.pdf', PLOTDIR, size), width=3.6, height=1.6,
                    pointsize=12, family='Helvetica')
par(mar=c(3.2, 3.5, 0.8, 0.7))
plot(1:nlay, x$entropy, col='black', lwd=3, type='l', bty='n', xlab='', ylab='',
     axes=FALSE, ylim=c(1,14))
axis(1, at=seq(0,nlay,20))
mtext('Layer', side=1, line=2)
axis(2, las=1) #, at=seq(0,15,5))
mtext('Entropy (bits)', side=2, line=2.4)
if (SAVE_PLOTS) dev.off()

# Token energy.
if (SAVE_PLOTS) pdf(sprintf('%s/token_energy_%s.pdf', PLOTDIR, size), width=3.6, height=1.6,
                    pointsize=12, family='Helvetica')
par(mar=c(3.2, 3.5, 0.8, 0.7))
plot(1:nlay, x$energy, col='#009E73', lwd=3, type='l', bty='n', xlab='', ylab='', axes=FALSE)
  #, ylim=c(0.22, 0.34))
axis(1, at=seq(0,nlay,20))
mtext('Layer', side=1, line=2)
# axis(2, las=1, at=seq(.22, .34, .04), labels=c('.22', '.26', '.30', '.34'))
axis(2, las=1, at=seq(.15, .3, .05), labels=c('.15', '.20', '.25', '.30'))
mtext('Token energy', side=2, line=2.4)
if (SAVE_PLOTS) dev.off()


##############################################################
### Procrustes analysis
##############################################################

# Compute optimal orthogonal matrices (rotations).
file <- sprintf('%s/data/RData/procrustes.RData', BASEDIR)
if (!file.exists(file)) {
  procrustes <- list()
  for (size in SIZES) {
    cat(sprintf('%s\n', size))
    U <- matrices$unemb[[size]]
    E <- matrices$emb[[size]]
    usv <- svd(U %*% t(E))
    # R is the optimal orthogonal matrix to align emb w/ unemb space.
    R <- usv$u %*% t(usv$v)
    procrustes[[size]] <- R
  }
  save(procrustes, file=file)
} else {
  load(file)
}

compute_ranks <- function(tok_pair, logits) {
  tok <- tok_pair[1]
  tok_en <- tok_pair[2]
  l <- logits[,tok]
  ranking <- names(l)[order(l, decreasing=TRUE)]
  c(which(ranking==tok), which(ranking==tok_en))
}

file <- sprintf('%s/data/RData/logit_ranks.RData', BASEDIR)
if (!file.exists(file)) {
  logits <- list()
  ranks <- list()
  relranks <- list()
  summs <- list()
  relsumms <- list()
  for (size in SIZES) {
    logits[[size]] <- list()
    ranks[[size]] <- list()
    relranks[[size]] <- list()
    relsumms[[size]] <- list()
    U <- matrices$unemb[[size]]
    E <- matrices$emb[[size]]
    R <- procrustes[[size]]
    for (lang in LANGS) {
      cat(sprintf('%s %s\n', size, lang))
      tok_pairs <- inputs[[lang]][,c('tok', 'tok_en')]
      # This is the actual computation of logits: rotate input embs, then project onto output embs.
      l <- t(U) %*% R %*% E[,tok_pairs$tok]
      logits[[size]][[lang]] <- l
      
      r <- t(apply(tok_pairs, 1,
                   function(tok_pairs) compute_ranks(tok_pairs, l)))
      rownames(r) <- tok_pairs$tok
      colnames(r) <- c(sprintf('rank_%s', lang), 'rank_en')
      rr <- (r-1)/(length(toks)-1)
      ranks[[size]][[lang]] <- r
      relranks[[size]][[lang]] <- rr
      
      summs[[size]][[lang]] <- summary(r)
      relsumms[[size]][[lang]] <- summary(rr)
    }
  }
  save(ranks, relranks, logits, summs, relsumms, file=file)
} else {
  load(file)
}

write_summs <- function(summs, file_prefix) {
  for (size in SIZES) {
    s <- do.call(cbind, summs[[size]])
    colnames(s) <- paste('rank', c('de_de', 'de_en',
                                   'fr_fr', 'fr_en',
                                   'ru_ru', 'ru_en'), sep='_')
    write.table(s, file=sprintf('%s/data/logit_ranks/%s_%s.tsv', BASEDIR, file_prefix, size),
                quote=FALSE, sep="\t", row.names=FALSE, col.names=TRUE)
  }
}

write_summs(summs, 'ranks')
write_summs(relsumms, 'relranks')


##############################################################
### MDS analysis
##############################################################

pair <- 'de_de'
size <- '13b'

# Load unemb matrix.
U <- matrices$unemb[[size]]

# Sample latent paths.
latents <- load_latents__OLD(size, pair)
nlay <- ncol(latents[[1]])
nL <- 50
set.seed(8)
idxL <- sample(length(latents), nL)
# RMS-normalize.
L <- rmsnorm(do.call(cbind, latents[idxL]))

# Sample tokens from U.
nU <- 4000
idxU <- rev(1:nU)
# Make sure to include German and English tokens for the sampled latent paths.
# NB: Some might be included multiple times.
path_tok_ids <- c(inputs$de$out_token_id[idxL], inputs$de$latent_token_id[idxL]) + 1
idxU[1:length(path_tok_ids)] <- path_tok_ids
UU <- U[,idxU]. # Unused?!

ProbUL <- compute_probs(U, L, idxU)
# Make latents unrelated (prob = 0); same for unemb vectors.
ZeroU <- matrix(0, nU, nU)
ZeroL <- matrix(0, nL*nlay, nL*nlay)
# Combined prob matrix.
S <- rbind(cbind(ZeroU, ProbUL),
           cbind(t(ProbUL), ZeroL))
# Smooth and take log.
Slog <- log(S + min(ProbUL))
# Turn similarity into distance.
D <- -Slog

# MDS.
mds <- cmdscale(D)

mdsU <- mds[1:nU,]
mdsL <- mds[(nU+1):(nU+nL*nlay),]

# All sampled paths.
if (SAVE_PLOTS) pdf(sprintf('%s/mds.pdf', PLOTDIR), width=3.4, height=2.8, pointsize=6,
                    family='Helvetica', useDingbats=FALSE)
par(mar=c(4, 4, 0.8, 0.8))

plot(mds[,1], mds[,2], col='white', bty='n',
     xlab='MDS dimension 1', ylab='MDS dimension 2',
     xlim=quantile(mds[,1],c(0.00,1)),
     ylim=quantile(mds[,2],c(0.00,1)))
points(mdsU[,1], mdsU[,2], col=rgb(0,0,0,0.1), pch=4)
points(mdsL[,1], mdsL[,2], col=rgb(0,0,0,0.4))

for (path in 1:nL){
  col <- rainbow(nlay, start=0, end=0.8, alpha=0.2)
  segments(x0=mdsL[((path-1)*nlay+1):(path*nlay-1),1],
           y0=mdsL[((path-1)*nlay+1):(path*nlay-1),2],
           x1=mdsL[((path-1)*nlay+2):(path*nlay),1],
           y1=mdsL[((path-1)*nlay+2):(path*nlay),2], col=col, lwd=2)
}

text(mdsU[1:nL,1], mdsU[1:nL,2], toks[path_tok_ids][1:nL], adj=0, col=COLS$de)
text(mdsU[(nL+1):(2*nL),1], mdsU[(nL+1):(2*nL),2], sub('^▁', '', toks[path_tok_ids][(nL+1):(2*nL)]), adj=0, col=COLS$en)

legend('topright', legend=c('Latent embeddings', 'Output token embeddings'),
       pch=c(1,4), bty='n', col=c('gray', 'gray'))

if (SAVE_PLOTS) dev.off()

# A single path.
plot(mds[,1], mds[,2], col='white',
     xlim=quantile(mds[,1],c(0.00,1)),
     ylim=quantile(mds[,2],c(0.00,1)))
points(mdsU[,1], mdsU[,2], col=rgb(0,0,1,0.1))
points(mdsL[,1], mdsL[,2], col=rgb(0,0,0,0.3))

path <- 13
col <- rainbow(nlay, start=0, end=0.8, alpha=1)
segments(x0=mdsL[((path-1)*nlay+1):(path*nlay-1),1],
         y0=mdsL[((path-1)*nlay+1):(path*nlay-1),2],
         x1=mdsL[((path-1)*nlay+2):(path*nlay),1],
         y1=mdsL[((path-1)*nlay+2):(path*nlay),2], col=col, lwd=2)

text(mdsU[path,1], mdsU[path,2], toks[path_tok_ids][path])
text(mdsU[nL+path,1], mdsU[nL+path,2], toks[path_tok_ids][nL+path], col='red')


##############################################################
### MDS analysis (DE -> ZH)
##############################################################

pair <- 'de_zh'
size <- '70b'

inputs_zh <- read.csv(sprintf('%s/data/inputs/clean/de_zh.csv', BASEDIR), header=FALSE,
                      col.names=c('out_token_id','out_token_str','latent_token_id',
                                  'latent_token_str','in_token_id','in_token_str'),
                      colClasses=c('integer', 'character'))
inputs_zh$tok_id <- inputs_zh$out_token_id + 1
inputs_zh$tok <- toks[inputs_zh$tok_id]
# The input file doesn't always contain the full English token; sometimes it
# contains a prefix (the original file contained all tokens that we match, but
# when preprocessing the file, I erroneously sometimes kept a prefix instead of
# the full token). Hence manually create a vector of full English tokens instead.
en_with_underscore <- sprintf('▁%s', inputs_zh$latent_token_str)
inputs_zh$tok_en_id <- tok_ids[en_with_underscore] + 1
inputs_zh$tok_en <- en_with_underscore
# One English full token ("pond") doesn't have a version starting with underscore.
# Fix it manually.
bad_idx <- which(is.na(inputs_zh$tok_en_id))
inputs_zh$tok_en[bad_idx] <- inputs_zh$latent_token_str[bad_idx]
inputs_zh$tok_en_id[bad_idx] <- tok_ids[inputs_zh$tok_en[bad_idx]] + 1

# Load unemb matrix.
U <- matrices$unemb[[size]]

# Sample latent paths.
latents <- load_latents(size, pair)
nlay <- ncol(latents[[1]])
nL <- 30
set.seed(9)
idxL <- sample(length(latents), nL)
# RMS-normalize.
L <- rmsnorm(do.call(cbind, latents[idxL]))

# Select tokens from U.
nU <- 2400
idxU <- rev(1:nU)
# Make sure to include German and English tokens for the selected latent paths.
path_tok_ids <- c(inputs_zh$tok_id[idxL], inputs_zh$tok_en_id[idxL])
idxU[1:length(path_tok_ids)] <- path_tok_ids

ProbUL <- compute_probs(U, L, idxU)
# Make latents unrelated (prob = 0); same for unemb vectors.
ZeroU <- matrix(0, nU, nU)
ZeroL <- matrix(0, nL*nlay, nL*nlay)
# Combined prob matrix.
S <- rbind(cbind(ZeroU, ProbUL),
           cbind(t(ProbUL), ZeroL))
# Smooth and take log.
Slog <- log(S + min(ProbUL))
# Turn similarity into distance.
D <- -Slog

# MDS.
mds <- cmdscale(D)

mdsU <- mds[1:nU,]
mdsL <- mds[(nU+1):(nU+nL*nlay),]

if (SAVE_PLOTS) cairo_pdf(sprintf('%s/mds.pdf', PLOTDIR), width=3.4, height=2.8,
                          pointsize=6, family='Heiti SC')
par(mar=c(4, 4, 0.1, 0.8))

plot(mds[,1], mds[,2], col='white', bty='n',
     xlab='MDS dimension 1', ylab='MDS dimension 2',
     xlim=c(-4,9),
     ylim=c(-6,4.2))
     # xlim=quantile(mds[,1],c(0.00,1)),
     # ylim=quantile(mds[,2],c(0.00,1)))
points(mdsU[,1], mdsU[,2], col=rgb(0,0,0,0.1), pch=4)
points(mdsL[,1], mdsL[,2], col=rgb(0,0,0,0.4))

for (path in 1:nL){
  col <- rainbow(nlay, start=0, end=0.8, alpha=0.2)
  segments(x0=mdsL[((path-1)*nlay+1):(path*nlay-1),1],
           y0=mdsL[((path-1)*nlay+1):(path*nlay-1),2],
           x1=mdsL[((path-1)*nlay+2):(path*nlay),1],
           y1=mdsL[((path-1)*nlay+2):(path*nlay),2], col=col, lwd=2)
}

labs_zh <- toks[path_tok_ids][1:nL]
labs_en <- sub('^▁', '', toks[path_tok_ids][(nL+1):(2*nL)])
text(mdsU[1:nL,1], mdsU[1:nL,2], labs_zh, adj=0, col=COLS$de)
text(mdsU[(nL+1):(2*nL),1], mdsU[(nL+1):(2*nL),2], labs_en, adj=0, col=COLS$en)

legend('topleft', legend=c('Latent embeddings', 'Token embeddings'),
       pch=c(1,4), col=c('gray', 'gray'), inset=c(0.02, 0.02))

if (SAVE_PLOTS) dev.off()

# A single path.
if (SAVE_PLOTS) cairo_pdf(sprintf('%s/mds_single.pdf', PLOTDIR), width=3.4, height=2.8,
                          pointsize=6, family='Heiti SC')
plot(mds[,1], mds[,2], col='white', bty='n',
     xlab='MDS dimension 1', ylab='MDS dimension 2',
     # xlim=c(min(mds[,1]), 8),
     # ylim=c(-4,4))
     xlim=quantile(mds[,1],c(0.00,1)),
     ylim=quantile(mds[,2],c(0.00,1)))
points(mdsU[,1], mdsU[,2], col=rgb(0,0,0,0.1), pch=4)
points(mdsL[,1], mdsL[,2], col=rgb(0,0,0,0.4))

path <- 4
col <- rainbow(nlay, start=0, end=0.8, alpha=1)
segments(x0=mdsL[((path-1)*nlay+1):(path*nlay-1),1],
         y0=mdsL[((path-1)*nlay+1):(path*nlay-1),2],
         x1=mdsL[((path-1)*nlay+2):(path*nlay),1],
         y1=mdsL[((path-1)*nlay+2):(path*nlay),2], col=col, lwd=2)

labs_zh <- toks[path_tok_ids][path]
labs_en <- sub('^▁', '', toks[path_tok_ids][nL+path])
text(mdsU[path,1], mdsU[path,2], labs_zh, adj=0, col=COLS$de)
text(mdsU[nL+path,1], mdsU[nL+path,2], labs_en, adj=0, col=COLS$en)

if (SAVE_PLOTS) dev.off()


##############################################################
# Energy analysis
##############################################################

# T: matrix of token embeddings (U or E)
# L: matrix of latents
compute_energy_ratios <- function(T, L, nlay) {
  # Normalize the columns of T and L to norm 1.
  T <- t(t(T) / sqrt(colSums(T^2)))
  L <- t(t(L) / sqrt(colSums(L^2)))
  # We're interested in ||T'*T||_F^2 = ||T*T'||_F^2 (the latter is faster to compute).
  # The squared Frobenius norm is the sum of squared projections of the columns
  # of T onto one another.
  TT <- T %*% t(T)
  # Now divide by the number of token pairs, obtaining the average, and take the
  # sqrt to get the average dot product between tokens. This serves as a baseline.
  avgTT <- sqrt(sum(TT^2) / ncol(T)^2)
  # For each layer, compute RMS norm of latent after projecting onto output tokens.
  norms <- sqrt(colMeans((t(T) %*% L)^2))
  # Compute ratio of norms and baseline.
  ratios <- norms / avgTT
  # Reshape such that we have one row per trajectory.
  ratios <- matrix(ratios, nrow=length(latents), ncol=nlay, byrow=TRUE)
  ratios
}

pair <- 'de_de'
size <- '13b'

# Load emb and unemb matrices.
U <- matrices$unemb[[size]]
E <- matrices$emb[[size]]
# Load latents.
latents <- load_latents(size, pair)
nlay <- ncol(latents[[1]])
L <- rmsnorm(do.call(cbind, latents))

ratios_UL <- compute_energy_ratios(U, L, nlay)
ratios_EL <- compute_energy_ratios(E, L, nlay)

matplot(1:nlay-1, t(ratios_UL), type='l', col=rgb(1,0,1,0.08), lty=1,
        xlab='Layer', ylab='Token energy', bty='n',
        panel.first=abline(v=seq(0,40,5), col='gray', lty=3))
lines(1:nlay-1, colMeans(ratios_UL), col='magenta', lwd=2)

matplot(1:nlay-1, t(ratios_EL), type='l', col=rgb(0,1,0,0.08), lty=1,
        xlab='Layer', ylab='Token energy', bty='n',
        panel.first=abline(v=seq(0,40,5), col='gray', lty=3))
lines(1:nlay-1, colMeans(ratios_EL), col='green', lwd=2)

# Only averages.
if (SAVE_PLOTS) pdf(sprintf('%s/token_energy.pdf', PLOTDIR), width=3.6, height=1.6,
                    pointsize=12, family='Helvetica')
par(mar=c(3.2, 3.5, 0.8, 0.7))
plot(1:nlay-1, colMeans(ratios_UL), col='#009E73', lwd=3, type='l', bty='n', xlab='', ylab='',
     axes=FALSE, ylim=c(0.05, 0.25))
# plot(1:nlay-1, sqrt(colMeans(ratios_UL^2)), col='#009E73', lwd=3, type='l', bty='n', xlab='', ylab='',
#      axes=FALSE, ylim=c(0.05, 0.25))
axis(1, at=seq(0,nlay,5))
mtext('Layer', side=1, line=2)
axis(2, las=1, at=seq(.05, .25, .05), labels=c('.05', '.10', '.15', '.20', '.25'))
mtext('Token energy', side=2, line=2.4)
if (SAVE_PLOTS) dev.off()


##############################################################
# Spectral analysis of token spaces.
##############################################################

size <- '70b'
U <- matrices$unemb[[size]]

# Center and compute SVD.
d <- svd(U-rowMeans(U), nu=0, nv=0)$d

relvar <- cumsum(d^2)/sum(d^2)
plot(relvar, type='l', panel.first=grid())

# 58% of PCs capture 80% of total variance
which(relvar>.8)[1]/nrow(U)


##############################################################
# Norm of U.
##############################################################

size <- '70b'
U <- matrices$unemb[[size]]
norms <- sqrt(colSums(U^2))

hist(norms, breaks='fd')
summary(norms)
sd(norms)
sd(norms)/mean(norms)

















#########################
# SCRATCH
#########################
