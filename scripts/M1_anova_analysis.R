# ============================================================
# One-way ANOVA + Tukey HSD for M1 values: WT vs ΔezrA vs ΔezrA-pCL55-ezrA
# ============================================================
# Packages: install.packages(c("readxl", "dplyr", "tidyr", "ggplot2", "car"))
library(readxl)
library(dplyr)
library(tidyr)
library(ggplot2)
library(car)      # leveneTest (Brown-Forsythe version)

# ---- 0. Settings --------------------------------------------------------
infile <- "M1_values.xlsx"



# ---- 1. Load data -------------------------------------------------------
raw <- read_excel(infile)
names(raw) <- c("WT", "dezrA", "comp")   # simple names; relabelled for plots below

if (REMOVE_DUPLICATES) {
  stopifnot(isTRUE(all.equal(raw$comp[38:54], raw$comp[21:37])))  # confirm duplication
  raw$comp[38:54] <- NA
}

lab <- c(WT = "WT", dezrA = "\u0394ezrA", comp = "\u0394ezrA-pCL55-ezrA")

dat <- raw %>%
  pivot_longer(everything(), names_to = "strain", values_to = "value") %>%
  filter(!is.na(value)) %>%
  mutate(strain = factor(strain, levels = c("WT", "dezrA", "comp")))

# ---- 2. Descriptive statistics -----------------------------------------
desc <- dat %>%
  group_by(strain) %>%
  summarise(n = n(), mean = mean(value), sd = sd(value),
            sem = sd / sqrt(n), median = median(value), .groups = "drop")
print(desc)

# ---- 3. Assumption checks ----------------------------------------------
fit <- aov(value ~ strain, data = dat)

cat("\n--- Brown-Forsythe (Levene, median-centred) ---\n")
print(leveneTest(value ~ strain, data = dat, center = median))

cat("\n--- Bartlett ---\n")
print(bartlett.test(value ~ strain, data = dat))

cat("\n--- Shapiro-Wilk per group ---\n")
print(dat %>% group_by(strain) %>%
        summarise(W = shapiro.test(value)$statistic,
                  p = shapiro.test(value)$p.value, .groups = "drop"))

cat("\n--- Shapiro-Wilk on ANOVA residuals ---\n")
print(shapiro.test(residuals(fit)))

# ---- 4. Ordinary one-way ANOVA -----------------------------------------
cat("\n--- One-way ANOVA ---\n")
print(summary(fit))

# ---- 5. Tukey HSD (Tukey-Kramer adjustment for unequal n) --------------
cat("\n--- Tukey HSD ---\n")
tuk <- TukeyHSD(fit, conf.level = 0.95)
print(tuk)

tuk_df <- as.data.frame(tuk$strain)
tuk_df$comparison <- rownames(tuk_df)
write.csv(tuk_df, "M1_tukey_results.csv", row.names = FALSE)

# ---- 6. Sensitivity analysis: non-parametric ---------------------------
# WT and complement fail normality (bounded at 1, left-skewed), so confirm
# the conclusions with Kruskal-Wallis + pairwise Wilcoxon (Holm-adjusted).
cat("\n--- Kruskal-Wallis ---\n")
print(kruskal.test(value ~ strain, data = dat))

cat("\n--- Pairwise Wilcoxon rank-sum, Holm-adjusted ---\n")
print(pairwise.wilcox.test(dat$value, dat$strain, p.adjust.method = "holm",
                           exact = FALSE))

# ---- 7. Figure ----------------------------------------------------------
p_to_stars <- function(p) {
  if (p < 0.0001) "****" else if (p < 0.001) "***" else
    if (p < 0.01) "**" else if (p < 0.05) "*" else "ns"
}

# TukeyHSD row names: "dezrA-WT", "comp-WT", "comp-dezrA"
p_wt_d  <- tuk_df["dezrA-WT",    "p adj"]
p_wt_c  <- tuk_df["comp-WT",     "p adj"]
p_d_c   <- tuk_df["comp-dezrA",  "p adj"]

bracket <- function(x1, x2, y, label) {
  list(
    annotate("segment", x = x1, xend = x2, y = y, yend = y),
    annotate("segment", x = x1, xend = x1, y = y - 0.015, yend = y),
    annotate("segment", x = x2, xend = x2, y = y - 0.015, yend = y),
    annotate("text", x = (x1 + x2) / 2, y = y + 0.02, label = label, size = 4)
  )
}

set.seed(1)
fig <- ggplot(dat, aes(strain, value)) +
  geom_jitter(aes(colour = strain), width = 0.15, size = 1.6, alpha = 0.6) +
  stat_summary(fun = mean, geom = "crossbar", width = 0.5,
               fatten = 1.5, colour = "black") +
  stat_summary(fun.data = function(x) {
    data.frame(y = mean(x), ymin = mean(x) - sd(x), ymax = mean(x) + sd(x))
  }, geom = "errorbar", width = 0.2, colour = "black") +
  bracket(1, 2, 1.06, p_to_stars(p_wt_d)) +
  bracket(2, 3, 1.15, p_to_stars(p_d_c)) +
  bracket(1, 3, 1.24, p_to_stars(p_wt_c)) +
  scale_x_discrete(labels = lab) +
  scale_colour_manual(values = c(WT = "#4D4D4D", dezrA = "#D55E00",
                                 comp = "#0072B2"), guide = "none") +
  scale_y_continuous(breaks = seq(0, 1, 0.2), limits = c(0, 1.3)) +
  labs(x = NULL, y = "M1 value") +   # <-- rename y-axis to the actual measurement
  theme_classic(base_size = 13) +
  theme(axis.text.x = element_text(face = "italic"))

ggsave("M1_figure.pdf", fig, width = 4.5, height = 4.5)
ggsave("M1_figure.png", fig, width = 4.5, height = 4.5, dpi = 300)
print(fig)
