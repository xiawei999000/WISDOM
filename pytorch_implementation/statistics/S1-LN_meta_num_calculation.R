# calculate LN meta number and cN stage
data_path <- './'
center_name <- 'FUDAN'  # SHANXI FUDAN  ZFY

patches_preds_path <- paste(data_path,center_name,'_patches_preds.xlsx', sep = "")
patches_preds_info <- read.xlsx(patches_preds_path, 1)

p_preds_path <- paste(data_path,center_name,'_patient_preds.xlsx', sep = "")
p_preds_info <- read.xlsx(p_preds_path, 1)

p_names <- p_preds_info$name
patches_names <- patches_preds_info$name

# meta(>cutoff) or not meta (<cutoff)
LN_preds_cutoff <- cutoff_meta(p_preds_info$meta_status, p_preds_info$patch_pred_max)

patches_preds <- patches_preds_info$patch_pred

# meta status for each patch
patches_pred_meta <- patches_preds
patches_pred_meta[which(patches_preds>LN_preds_cutoff)] <- 1
patches_pred_meta[which(patches_preds<LN_preds_cutoff)] <- 0

# the predicted LN meta num for each patient
LN_meta_num_pred_p <- LN_meta_num(p_names, patches_names, patches_pred_meta)
cNstage <- cN_stage(LN_meta_num_pred_p)
LN_num_info_merge <- cbind(p_preds_info, LN_meta_num_pred_p, cNstage)

result_path <-paste(data_path,center_name,'_patient_preds_LN_num.xlsx', sep = "")
write.xlsx(LN_num_info_merge, result_path, row.names = F)
