#############################################################
# Create edx set, validation set, and submission file
#############################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                      col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

###########################################################################
## Declare a pair of functions that we'll need later
###########################################################################

# RMSE function
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Predict ratings
predict_ratings <- function(dataset, mu, b_i, b_u, g_ui) {
  dataset %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(g_ui, by = c("userId","genres")) %>%
    replace_na(list(b_i=0,b_u=0, g_ui=0)) %>%
    mutate(pred = mu + b_i + b_u + g_ui) %>%
    pull(pred)
}

###########################################################################
## Here start the tuning of the parameter lambda used for regularization
###########################################################################

set.seed(1)
# Divide the edx data frame in two sets:
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.25, list = FALSE)
tuning_set <- edx[-test_index,]
test_set <- edx[test_index,]

rm(test_index)

# Define the sequence of lambdas that will be examined
lambdas <- seq(0, 10, 0.25)

# Create folds for cross validation (5 folds)
folds <- createFolds(tuning_set$rating, k = 5, list = TRUE, returnTrain = FALSE)

# for each fold
rmses <- sapply(folds, function(cur_fold) {
  print(paste(Sys.time(), " - new fold "))
    
  cur_train <- tuning_set[-cur_fold,]
  cur_test <- tuning_set[cur_fold,]
  
  # First calculate the terms with no normalization to use them with the current fold  
  mu <- mean(cur_train$rating)
  
  # Movie effect
  b_i <- cur_train %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/n(), n_i = n())
  
  # cur_join: next left join will be needed more than once. Store it to not repeat
  cur_join <- cur_train %>% 
    left_join(b_i, by="movieId")
  
  # User effect  
  b_u <- cur_join %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/n(), n_u = n())
  
  cur_join <- cur_join %>%
    left_join(b_u, by="userId")
  
  # User genre preferences effect  
  g_ui <- cur_join %>%
    group_by(userId, genres) %>%
    summarize(g_ui = sum(rating - b_i - b_u - mu)/n(), n_g = n())
    
  rm(cur_join, cur_train)
  
  # cur_test: keep the left joins to not repeat this operation with each lambda
  # note: replace_na: When there is no data, I assume no effect
  cur_test <- cur_test %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(g_ui, by = c("userId","genres")) %>%
    replace_na(list(b_i=0,b_u=0, g_ui=0, n_i=0.000001, n_u=0.000001, n_g=0.000001))
  
  # For each lambda, calculate RMSE
  errors <- sapply(lambdas, function(l){
    print(paste(Sys.time(), " - Lambda ", l))
    
    predicted_ratings <- 
      cur_test %>%
      mutate(
        b_i = b_i * n_i /(n_i + l),
        b_u = b_u * n_u /(n_u + l),
        g_ui = g_ui * n_g /(n_g + l),
        pred = mu + b_i + b_u + g_ui) %>%
      pull(pred)
    
    RMSE(predicted_ratings, cur_test$rating)
  })

  return(errors)
})

rm(folds)

(rmses <- rowMeans(rmses))
qplot(lambdas, rmses)  

# Get the optimal lambda:
lambda <- lambdas[which.min(rmses)]

####### Evaluation of the best lambda with test set
mu <- mean(tuning_set$rating)

# Movie effect
b_i <- tuning_set %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda))

# cur_join: next left join will be needed more than once. Store it to not repeat
cur_join <- tuning_set %>% 
  left_join(b_i, by="movieId")

# User effect  
b_u <- cur_join %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))

cur_join <- cur_join %>%
  left_join(b_u, by="userId")

# User genre preferences effect  
g_ui <- cur_join %>%
  group_by(userId, genres) %>%
  summarize(g_ui = sum(rating - b_i - b_u - mu)/(n()+lambda))

rm(cur_join)

predicted_ratings <- predict_ratings(test_set, mu, b_i, b_u, g_ui)

result = RMSE(predicted_ratings, test_set$rating)
print(paste("The RMSE after tuning the lambda parameter is",result))

###########################################################################
## Using the optimal lambda with the hole edx data frame
###########################################################################

## remove some not needed data
rm(tuning_set, test_set, b_i, b_u, g_ui, predicted_ratings, lambdas, rmses)

# mean of the ratings of the hole edx data frame
mu <- mean(edx$rating)

# Movie effect
b_i <- edx %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda))

cur_join <- edx %>% 
  left_join(b_i, by="movieId")

# User effect  
b_u <- cur_join %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))

cur_join <- cur_join %>%
  left_join(b_u, by="userId")

# User genre preferences effect  
g_ui <- cur_join %>%
  group_by(userId, genres) %>%
  summarize(g_ui = sum(rating - b_i - b_u - mu)/(n()+lambda))

rm(cur_join)

###########################################################################
## Final result with validation data frame
###########################################################################
predicted_ratings <- predict_ratings(validation, mu, b_i, b_u, g_ui)

result = RMSE(predicted_ratings, validation$rating)

print("The final RMSE on the validation data frame is:")
result
