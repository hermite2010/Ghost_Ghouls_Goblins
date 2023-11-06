library(tidymodels)
library(embed)     # For target encoding
library(vroom) 
library(parallel)
library(ggmosaic)  # For plotting
library(ranger)    # FOR RANDOM/CLASSIFICATION FOREST
library(doParallel)
library(discrim)
library(naivebayes)# FOR NAIVE BAYES
library(kknn)
library(themis)    # for smote

# Reading in the Data
GGG_Train <- vroom("Ghost_Ghouls_Goblins/train.csv") #"Amazon_AEAC_Kaggle/train.csv" for local
GGG_Test <- vroom("Ghost_Ghouls_Goblins/test.csv") #"Amazon_AEAC_Kaggle/test.csv" for local

GGG_recipe <- recipe(type ~., data=GGG_Train) %>%
#  step_mutate_at(color, fn = factor) %>% 
  step_lencode_glm(color, outcome = vars(type)) %>%
#  update_role(id, new_role="id")
  #step_smote(all_numeric_predictors(), neighbors = 5) #%>% 

prepped_recipe <- prep(GGG_recipe)
baked_data <- bake(prepped_recipe, new_data=GGG_Train)

# Accounting for missing values -------------------------------------------

# GGG_missing <- vroom('Ghost_Ghouls_Goblins/trainWithMissingValues.csv')
# 
# head(GGG_missing)
# 
# GGG_missing_recipe <- recipe(type ~., data=GGG_missing) %>% 
#   step_impute_bag(bone_length, impute_with = imp_vars(hair_length, rotting_flesh, has_soul, color), trees = 100) %>% 
#   step_impute_bag(hair_length, impute_with = imp_vars(bone_length, rotting_flesh, has_soul, color), trees = 100) %>%
#   step_impute_bag(rotting_flesh, impute_with = imp_vars(hair_length, bone_length, has_soul, color), trees = 100) %>%
#   step_impute_bag(has_soul, impute_with = imp_vars(hair_length, rotting_flesh, bone_length, color), trees = 100) %>%
#   step_impute_mode(color)
# 
# GGG_missing_recipe <- recipe(type ~., data=GGG_missing) %>% 
#   step_impute_bag(all_numeric_predictors(), impute_with = imp_vars(all_predictors()), trees = 10)
# 
# # apply the recipe to your data
# prepped_recipe <- prep(GGG_missing_recipe)
# baked_data <- bake(prepped_recipe, new_data=GGG_missing)
# 
# rmse_vec(GGG_Train[is.na(GGG_missing)], baked_data[is.na(GGG_missing)])

  
# Multinomial Random Forest -----------------------------------------------
# Set Up the Engine
mult_for_mod <- rand_forest(mtry = tune(),
                             min_n=tune(),
                             trees=500) %>% #Type of model
  set_engine("ranger") %>% # What R function to use
  set_mode("classification")

## Workflow and model and recipe
mult_for_wf <- workflow() %>%
  add_recipe(GGG_recipe) %>%
  add_model(mult_for_mod)

## set up grid of tuning values
mult_tuning_grid <- grid_regular(mtry(range = c(1,(ncol(GGG_Train)-1))),
                                  min_n(),
                                  levels = 6)

## set up k-fold CV
mult_folds <- vfold_cv(GGG_Train, v = 4, repeats=1)

# ## Set up Parallel processing
# num_cores <- as.numeric(parallel::detectCores())#How many cores do I have?
# if (num_cores > 4)
#   num_cores = 10
# cl <- makePSOCKcluster(num_cores)
# registerDoParallel(cl)

## Run the CV
CV_results <- mult_for_wf %>%
  tune_grid(resamples=mult_folds,
            grid=mult_tuning_grid,
            metrics=metric_set(accuracy)) #Or leave metrics NULL

#stopCluster(cl)
## find best tuning parameters
bestTune <- CV_results %>%
  select_best("accuracy")

## Finalize workflow and prediction 
final_wf <- mult_for_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=GGG_Train)

class_for_preds <- final_wf %>%
  predict(new_data = GGG_Test, type="class") %>% 
  mutate(id = GGG_Test$id) %>% 
  rename(type = .pred_class) %>% 
  select(2,1)

## Write it out
vroom_write(x=class_for_preds, file="MultForest.csv", delim=",")


# Naive Bayes -------------------------------------------------------------
## Set Up Model
nb_recipe <- recipe(type ~., data=GGG_Train) %>% 
  step_lencode_glm(color, outcome = vars(type))

nb_model <- naive_Bayes(Laplace=tune(), 
                        smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes")

## Workflow and model and recipe
nb_wf <- workflow() %>%
  add_recipe(nb_recipe) %>%
  add_model(nb_model)

## set up grid of tuning values
nb_tuning_grid <- grid_regular(Laplace(),
                               smoothness(),
                               levels = 10)

## set up k-fold CV
nb_folds <- vfold_cv(GGG_Train, v = 5, repeats=1)

# ## Set up Parallel processing
# num_cores <- as.numeric(parallel::detectCores())#How many cores do I have?
# if (num_cores > 4)
#   num_cores = 10
# cl <- makePSOCKcluster(num_cores)
# registerDoParallel(cl)

## Run the CV
CV_results <- nb_wf %>%
  tune_grid(resamples=nb_folds,
            grid=nb_tuning_grid,
            metrics=metric_set(accuracy)) #Or leave metrics NULL
#stopCluster(cl)

## find best tuning parameters
bestTune <- CV_results %>%
  select_best("accuracy")

## Finalize workflow and prediction 
final_wf <- nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=GGG_Train)

nb_preds <- final_wf %>%
  predict(new_data = GGG_Test, type="class") %>% 
  mutate(id = GGG_Test$id) %>% 
  rename(type = .pred_class) %>% 
  select(2,1)

## Write it out
vroom_write(x=nb_preds, file="Ghost_Ghouls_Goblins/NaiveBayes.csv", delim=",")

# KNN ---------------------------------------------------------------------
## Set up model
knn_model <- nearest_neighbor(neighbors=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kknn")

## Set the Workflow
knn_wf <- workflow() %>%
  add_recipe(GGG_recipe) %>%
  add_model(knn_model)

## set up grid of tuning values
knn_tuning_grid <- grid_regular(neighbors(),
                                levels = 10)

## set up k-fold CV
knn_folds <- vfold_cv(GGG_Train, v = 10, repeats=1)

## Set up Parallel processing
# num_cores <- as.numeric(parallel::detectCores())#How many cores do I have?
# if (num_cores > 4)
#   num_cores = 10
# cl <- makePSOCKcluster(num_cores)
# registerDoParallel(cl)

## Run the CV
CV_results <- knn_wf %>%
  tune_grid(resamples=knn_folds,
            grid=knn_tuning_grid,
            metrics=metric_set(accuracy)) #Or leave metrics NULL

#stopCluster(cl)
## find best tuning parameters
bestTune <- CV_results %>%
  select_best("accuracy")

## Finalize workflow and prediction 
final_wf <- knn_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=GGG_Train)

knn_preds <- final_wf %>%
  predict(new_data = GGG_Test, type="class") %>% 
  mutate(id = GGG_Test$id) %>% 
  rename(type = .pred_class) %>% 
  select(2,1)

#knn_preds <- as.data.frame(cbind(GGG_Test$id), as.character(knn_preds$.pred_class))

## Write it out
vroom_write(x=knn_preds, file="Ghost_Ghouls_Goblins/KNN.csv", delim=",") #"Amazon_AEAC_Kaggle/KNN.csv"


# ANN ---------------------------------------------------------------------

nn_recipe <- recipe(type ~., data=GGG_Train) %>%
  update_role(id, new_role="id") %>%
  step_mutate_at(color, fn = factor) %>% 
  step_lencode_glm(color, outcome = vars(type)) %>% ## Turn color to factor then dummy encode color
  step_range(all_numeric_predictors(), min=0, max=1) #scale to [0,1]

nn_model <- mlp(hidden_units = tune(),
                epochs = 250) %>% #or 100 or 250
  set_engine("nnet") %>% #verbose = 0 prints off less
  set_mode("classification")

## Set the Workflow
nn_wf <- workflow() %>%
  add_recipe(nn_recipe) %>%
  add_model(nn_model)

nn_tuneGrid <- grid_regular(hidden_units(range=c(1, 75)),
                            levels=10)

## set up k-fold CV
nn_folds <- vfold_cv(GGG_Train, v = 10, repeats=1)

## Run the CV
tuned_nn <- nn_wf %>%
  tune_grid(resamples=nn_folds,
            grid=nn_tuneGrid,
            metrics=metric_set(accuracy)) #Or leave metrics NULL

tuned_nn %>% 
  collect_metrics() %>%
  filter(.metric=="accuracy") %>%
  ggplot(aes(x=hidden_units, y=mean)) + geom_line()

## find best tuning parameters
bestTune <-tuned_nn %>%
  select_best("accuracy")

## Finalize workflow and prediction 
final_wf <- nn_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=GGG_Train)

nn_preds <- final_wf %>%
  predict(new_data = GGG_Test, type="class") %>% 
  mutate(id = GGG_Test$id) %>% 
  rename(type = .pred_class) %>% 
  select(2,1)

## Write it out
vroom_write(x=nn_preds, file="Ghost_Ghouls_Goblins/NN_nnet.csv", delim=",")
