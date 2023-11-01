library(tidymodels)
library(embed)     # For target encoding
library(vroom) 
library(parallel)
library(ggmosaic)  # For plotting
library(ranger)    # FOR RANDOM/CLASSIFICATION FOREST
library(doParallel)
library(discrim)   # FOR NAIVE BAYES
library(kknn)
library(themis)    # for smote

# Reading in the Data
GGG_Train <- vroom("Ghost_Ghouls_Goblins/train.csv") #"Amazon_AEAC_Kaggle/train.csv" for local
GGG_Test <- vroom("Ghost_Ghouls_Goblins/test.csv") #"Amazon_AEAC_Kaggle/test.csv" for local

GGG_missing <- vroom('Ghost_Ghouls_Goblins/trainWithMissingValues.csv')

head(GGG_missing)

GGG_missing_recipe <- recipe(type ~., data=GGG_missing) %>% 
  step_impute_bag(bone_length, impute_with = imp_vars(hair_length, rotting_flesh, has_soul, color), trees = 100) %>% 
  step_impute_bag(hair_length, impute_with = imp_vars(bone_length, rotting_flesh, has_soul, color), trees = 100) %>%
  step_impute_bag(rotting_flesh, impute_with = imp_vars(hair_length, bone_length, has_soul, color), trees = 100) %>%
  step_impute_bag(has_soul, impute_with = imp_vars(hair_length, rotting_flesh, bone_length, color), trees = 100) %>%
  step_impute_mode(color)

GGG_missing_recipe <- recipe(type ~., data=GGG_missing) %>% 
  step_impute_bag(all_numeric_predictors(), impute_with = imp_vars(all_predictors()), trees = 10)

# apply the recipe to your data
prepped_recipe <- prep(GGG_missing_recipe)
baked_data <- bake(prepped_recipe, new_data=GGG_missing)

rmse_vec(GGG_Train[is.na(GGG_missing)], baked_data[is.na(GGG_missing)])

  