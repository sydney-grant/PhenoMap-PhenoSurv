install.packages("dplyr")
install.packages("plyr")
install.packages("doMC")
install.packages("randomForest")
install.packages("xgboost")

library(dplyr)
library(plyr)
library(doMC)
library(randomForest)
library(xgboost)


args <- commandArgs(trailingOnly = TRUE)

split_num <- args[[1]]
print(split_num)

print("Loading data...")
mutations <- read.csv("PANCAN_Mutations_Genelevel.csv")
mutations <- mutations[,-1]

cna <- read.csv("PANCAN_CNA.csv")
cna <- cna[,-1]

pathways <- read.csv("GSVA_Results_PANCAN.csv")
pathways <- pathways[grepl("KEGG", pathways[,1]),]


genes_pathways_hallmark <- read.csv(paste0("kegg_genefilter_", split_num, "_IMPACT.csv"))[,-1]


pathway_names <- pathways[,1]
pathways <- pathways[,-1]

print("Configuring ids...")
pathway_colnames <- c()
for (path in colnames(pathways)){
  new_path <- gsub("[.]", "-", path)
  new_path <- substr(new_path, 1, 12)
  pathway_colnames <- c(pathway_colnames, new_path)
}

new_ids <- c()
for (path in mutations$PatientID){
  new_path <- substr(path, 1, 12)
  new_ids <- c(new_ids, new_path)
}

colnames(pathways) <- pathway_colnames
mutations$PatientID <- unlist(new_ids)

overlap_patients <- intersect(cna$PatientID, mutations$PatientID)
pathways <- pathways[,intersect(overlap_patients, colnames(pathways))]

mutations <- subset(mutations, PatientID %in% intersect(overlap_patients, colnames(pathways)))
cna <- subset(cna, PatientID %in% intersect(overlap_patients, colnames(pathways)))

## remove patients with multiple samples
multiple_samples <- as.data.frame(table(mutations$PatientID))
multiple_samples <- subset(multiple_samples, Freq > 1)$Var1

mutations_fix <- subset(mutations, PatientID %in% multiple_samples)
mutations_keep <- dplyr::setdiff(mutations, mutations_fix)

mutations_fixed <- data.frame()
for (pt in multiple_samples){
  sub <- subset(mutations_fix, PatientID == pt)
  mutations_fixed <- rbind(mutations_fixed, sub[1,])
}

mutations <- rbind(mutations_keep, mutations_fixed)

mutations <- mutations %>%
  arrange(PatientID, colnames(pathways))
cna <- cna %>%
  arrange(PatientID, colnames(pathways))
pathways <- pathways[,cna$PatientID]
set.seed(1)

split1 <- sample(1:nrow(mutations), 0.2*nrow(mutations))
split2 <- sample(setdiff(1:nrow(mutations),split1), 0.2*nrow(mutations))
split3 <- sample(setdiff(1:nrow(mutations),c(split1, split2)), 0.2*nrow(mutations))
split4 <- sample(setdiff(1:nrow(mutations),c(split1, split2, split3)), 0.2*nrow(mutations))
split5 <- setdiff(1:nrow(mutations),c(split1, split2, split3, split4))

if (split_num == "split1"){split <- split1}
if (split_num == "split2"){split <- split2}
if (split_num == "split3"){split <- split3}
if (split_num == "split4"){split <- split4}
if (split_num == "split5"){split <- split5}

calc_pathway <- function(pathway, input_mutations, input_cna, train_indis, gene_df, model){
  pathname <- pathway[,length(pathway)]

  keep_genes <- subset(gene_df, Pathway == pathname)$Genes
  input_mutations <- input_mutations[,intersect(keep_genes, colnames(input_mutations))]
  input_cna <- input_cna[,intersect(keep_genes, colnames(input_cna))]

  
  colnames(input_mutations) <- paste0("SNV_", colnames(input_mutations))
  colnames(input_cna) <- paste0("CNA_", colnames(input_cna))
  input_data <- cbind(input_mutations, input_cna)
  
  x_train <- input_data[-train_indis,]
  x_test <- input_data[train_indis,]
  y_train <- pathway[-train_indis]
  y_test <- pathway[train_indis]
  
  y_train <- y_train[-ncol(y_train)]
  y_train <- unlist(as.list(y_train))
  y_test <- unlist(as.list(y_test))
  

  mod <- randomForest(x_train, y_train, mtry = sqrt(ncol(x_train)), type = "regression")
  y_predicted_train <- as.numeric(predict(mod, x_train))
  y_predicted_test <- as.numeric(predict(mod, x_test))
  train_r2 <- cor(as.numeric(y_predicted_train), as.numeric(y_train))
  test_r2 <- cor(as.numeric(y_predicted_test), as.numeric(y_test))
  outputs <- list("train_preds" = y_predicted_train, "test_preds" = y_predicted_test, "model" = mod)

  return(outputs)
}


pathways_t <- as.data.frame(t(pathways))
real_pathway_train <- pathways_t[-split,]
real_pathway_test <- pathways_t[split,]

real_mutation_train <- mutations[-split,]
real_mutation_test <- mutations[split,]

pt_ids <- mutations$PatientID

ids_train <- pt_ids[-split]
ids_test <- pt_ids[split]

print("Start pathway predictions...")
pathways$pathname <- as.factor(pathway_names)
doMC::registerDoMC(cores=16)

predicted_pathways <- dlply(pathways, .(pathname), calc_pathway, input_mutations = mutations[,-ncol(mutations)],
                            input_cna = cna[,-1],
                            train_indis = split,  gene_df = genes_pathways_hallmark,
                            .parallel = TRUE)



print("Pathway predictions complete.")
train_preds <- data.frame("UUID" = ids_train)
test_preds <- data.frame("UUID" = ids_test)
model_list <- list()
i <- 1
for (pathway in predicted_pathways){
  train_preds <- cbind(train_preds, pathway$train_preds)
  test_preds <- cbind(test_preds, pathway$test_preds)
   model_list[[i]] <- pathway$model
  i <- i + 1
}





print("Saving data...")
write.csv(train_preds, paste0("train_predictions_FINAL_", split_num, "_kegg_IMPACT.csv"))
write.csv(test_preds, paste0("test_predictions_FINAL_", split_num, "_kegg_IMPACT.csv"))
#saveRDS(model_list, "kegg_trained_models_IMPACT.rds")




