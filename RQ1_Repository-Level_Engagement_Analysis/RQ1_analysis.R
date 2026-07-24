options(repos = c(CRAN = "https://mirrors.tuna.tsinghua.edu.cn/CRAN/"))
options(scipen = 999)

packages_needed <- c("tidyverse", "lme4", "lmerTest", "MuMIn", "patchwork", "readr", "stringr", "car")
packages_to_install <- packages_needed[!(packages_needed %in% installed.packages()[,"Package"])]

if(length(packages_to_install) > 0) {
  install.packages(packages_to_install)
}

library(tidyverse)
library(lme4)
library(lmerTest)
library(MuMIn)
library(patchwork)
library(car)

LABEL_GROUP_1 <- "GDNP"
LABEL_GROUP_2 <- "Non-GDNP"

theme_academic <- function() {
  theme_classic(base_size = 18) +
    theme(
      text = element_text(family = "serif"),
      axis.line = element_line(linewidth = 0.8, color = "black"),
      axis.text = element_text(color = "black"),
      axis.title = element_text(face = "bold"),
      panel.grid.major.y = element_line(color = "grey90", linetype = "dashed"),
      plot.title = element_text(hjust = 0.5, face = "bold", size = 20),
      legend.position = "bottom",
      legend.box = "horizontal",
      legend.margin = margin(t = 5),
      legend.text = element_text(size = 14),
      legend.title = element_text(face="bold", size = 14)
    )
}

plot_rdd_comparison <- function(data, y_var, y_label_expr, title_text) {
  df_summary <- data %>%
    group_by(time_index, D_it, Group) %>%
    summarise(
      mean_y = mean(.data[[y_var]], na.rm = TRUE),
      se_y   = sd(.data[[y_var]], na.rm = TRUE) / sqrt(n()),
      .groups = "drop"
    )
  
  ggplot(data, aes(x = time_index, y = .data[[y_var]])) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "grey60", linewidth = 0.8) +
    geom_smooth(method = "lm", formula = y ~ x, 
                aes(color = factor(D_it), 
                    fill = factor(D_it), 
                    linetype = Group), 
                alpha = 0.1, linewidth = 1.2) +
    geom_point(data = df_summary, 
               aes(y = mean_y, color = factor(D_it), shape = Group), 
               size = 3, alpha = 0.9) +
    geom_errorbar(data = df_summary, 
                  aes(y = mean_y, ymin = mean_y - se_y, ymax = mean_y + se_y, 
                      color = factor(D_it)), 
                  width = 0.3, alpha = 0.6) +
    scale_color_manual(name = "Period:", 
                       values = c("0" = "#2E86C1", "1" = "#C0392B"), 
                       labels = c("Before Deprecation", "After Deprecation")) +
    scale_fill_manual(name = "Period:", 
                      values = c("0" = "#2E86C1", "1" = "#C0392B"), 
                      labels = c("Before Deprecation", "After Deprecation")) +
    scale_linetype_manual(name = "Data Source:", values = c(1, 2)) +
    scale_shape_manual(name = "Data Source:", values = c(16, 17)) +
    scale_x_continuous(breaks = seq(-12, 12, 3), name = "Months Relative to Deprecation") +
    ylab(y_label_expr) + 
    ggtitle(title_text) +
    theme_academic() +
    guides(
      color = guide_legend(order = 1, override.aes = list(shape = 15, linetype = 0, size = 5)),
      fill  = guide_legend(order = 1, override.aes = list(shape = 15, linetype = 0, size = 5)),
      shape = guide_legend(order = 2, override.aes = list(linetype = 0, size = 4, color = "black")),
      linetype = "none"
    )
}

cat("Loading datasets...\n")
raw1 <- read_csv("GDNP_data.csv", show_col_types = FALSE)
raw2 <- read_csv("Non_GDNP_data.csv", show_col_types = FALSE)

extract_long <- function(data, pattern, val_name) {
  data %>%
    select(package_name, matches(pattern)) %>%
    pivot_longer(-package_name, names_to = "temp_col", values_to = val_name) %>%
    mutate(time_index = as.numeric(str_extract(temp_col, "-?\\d+"))) %>%
    select(-temp_col)
}

process_group <- function(raw_data, group_label) {
  df_star     <- extract_long(raw_data, "^star_win_", "star")
  df_fork     <- extract_long(raw_data, "^fork_win_", "fork")
  df_issue    <- extract_long(raw_data, "_issue_created$", "issue_created")
  df_pr       <- extract_long(raw_data, "_pr_created$", "pr_created")
  df_download <- extract_long(raw_data, "^download_win_", "download")
  
  df_static <- raw_data %>%
    select(package_name, `Dependency count`, `Package Age`, `Contributors Count`, `Total Releases`)
  
  df_panel <- df_star %>%
    left_join(df_fork, by = c("package_name", "time_index")) %>%
    left_join(df_issue, by = c("package_name", "time_index")) %>%
    left_join(df_pr, by = c("package_name", "time_index")) %>%
    left_join(df_download, by = c("package_name", "time_index")) %>%
    left_join(df_static, by = "package_name") %>%
    mutate(Group = group_label)
  
  return(df_panel)
}

df1_all <- process_group(raw1, LABEL_GROUP_1)
df2_all <- process_group(raw2, LABEL_GROUP_2)
df_combined_all <- bind_rows(df1_all, df2_all)

valid_pkgs_sf <- df_combined_all %>%
  group_by(Group, package_name) %>%
  summarise(total_activity = sum(star, na.rm=TRUE) + sum(fork, na.rm=TRUE), .groups="drop") %>%
  filter(total_activity > 0)

df_model_sf <- df_combined_all %>%
  semi_join(valid_pkgs_sf, by = c("Group", "package_name")) %>%
  filter(time_index != 0) %>%
  mutate(
    T_it = time_index,
    D_it = ifelse(time_index > 0, 1, 0),
    P_it = ifelse(D_it == 1, time_index, 0),
    log_downloads    = log1p(download),
    log_dependencies = log1p(`Dependency count`),
    log_age          = log1p(`Package Age`),
    log_contributors = log1p(`Contributors Count`),
    log_releases     = log1p(`Total Releases`),
    log_star_growth  = log1p(star),
    log_fork_growth  = log1p(fork)
  )

valid_pkgs_ip <- df_combined_all %>%
  group_by(Group, package_name) %>%
  summarise(total_act = sum(issue_created, na.rm=TRUE) + sum(pr_created, na.rm=TRUE), .groups="drop") %>%
  filter(total_act > 0)

df_model_ip <- df_combined_all %>%
  semi_join(valid_pkgs_ip, by = c("Group", "package_name")) %>%
  filter(time_index != 0) %>%
  mutate(
    T_it = time_index,
    D_it = ifelse(time_index > 0, 1, 0),
    P_it = ifelse(D_it == 1, time_index, 0),
    log_downloads    = log1p(download),
    log_dependencies = log1p(`Dependency count`),
    log_age          = log1p(`Package Age`),
    log_contributors = log1p(`Contributors Count`),
    log_releases     = log1p(`Total Releases`),
    log_issue_created = log1p(issue_created),
    log_pr_created    = log1p(pr_created)
  )

generate_table_2_results <- function(data, group_name, outcome_var, outcome_name) {
  subset_data <- data %>% filter(Group == group_name)
  
  formula_str <- paste(outcome_var, 
                       "~ T_it + D_it + P_it + log_downloads + log_dependencies + log_age + log_contributors + log_releases + (1 | package_name)")
  
  cat(paste0("\n=======================================================\n"))
  cat(paste0(">>> Output for Table 2: ", outcome_name, " | ", group_name, " <<<\n"))
  cat(paste0("=======================================================\n"))
  
  model <- lmer(as.formula(formula_str), data = subset_data, REML = TRUE)
  
  print(round(coef(summary(model)), 3))
  
  vif_values <- car::vif(model)
  print(round(vif_values, 2))
  
  control_vars <- c("log_downloads", "log_dependencies", "log_age", "log_contributors", "log_releases")
  vif_controls <- vif_values[names(vif_values) %in% control_vars]
  
  if(any(vif_controls > 3)) {
    cat("WARNING: Some control variables have VIF > 3.\n")
  } else {
    cat("VIF check passed: All control variables have VIF <= 3.\n")
  }
  
  r2_vals <- r.squaredGLMM(model)
  cat("\n--- Model Fits ---\n")
  cat(sprintf("Marginal R2 (Rm2)    : %.3f\n", r2_vals[1, "R2m"]))
  cat(sprintf("Conditional R2 (Rc2) : %.3f\n", r2_vals[1, "R2c"]))
  
  return(model)
}

cat("\nFitting RDD Models to generate Table 2 parameters...\n")

invisible(generate_table_2_results(df_model_sf, LABEL_GROUP_1, "log_star_growth", "New Stars"))
invisible(generate_table_2_results(df_model_sf, LABEL_GROUP_2, "log_star_growth", "New Stars"))
invisible(generate_table_2_results(df_model_sf, LABEL_GROUP_1, "log_fork_growth", "New Forks"))
invisible(generate_table_2_results(df_model_sf, LABEL_GROUP_2, "log_fork_growth", "New Forks"))

invisible(generate_table_2_results(df_model_ip, LABEL_GROUP_1, "log_issue_created", "New Issues"))
invisible(generate_table_2_results(df_model_ip, LABEL_GROUP_2, "log_issue_created", "New Issues"))
invisible(generate_table_2_results(df_model_ip, LABEL_GROUP_1, "log_pr_created", "New PRs"))
invisible(generate_table_2_results(df_model_ip, LABEL_GROUP_2, "log_pr_created", "New PRs"))

cat("\n=======================================================\n")
cat("Generating plots and saving to disk...\n")

p_stars <- plot_rdd_comparison(df_model_sf, "log_star_growth", expression(bold(ln("New Stars" + 1))), "A. New Stars")
p_forks <- plot_rdd_comparison(df_model_sf, "log_fork_growth", expression(bold(ln("New Forks" + 1))), "B. New Forks")
p_issues <- plot_rdd_comparison(df_model_ip, "log_issue_created", expression(bold(ln("New Issues" + 1))), "C. New Issues")
p_prs <- plot_rdd_comparison(df_model_ip, "log_pr_created", expression(bold(ln("New PRs" + 1))), "D. New Pull Requests")

final_plot_combined <- (p_stars + p_forks + p_issues + p_prs) + 
  plot_layout(nrow = 1, guides = "collect") & 
  theme(legend.position = "bottom",
        legend.direction = "horizontal",
        legend.box = "horizontal")

print(final_plot_combined)

ggsave("RQ1_result.png", final_plot_combined, width = 24, height = 7, dpi = 300)
ggsave("RQ1_result.pdf", final_plot_combined, width = 24, height = 7, dpi = 300, device = cairo_pdf)

cat("Process Complete.\n")