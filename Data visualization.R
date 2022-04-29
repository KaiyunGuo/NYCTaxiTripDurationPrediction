library(lubridate)
library(geosphere)
library(tidyverse)
library(ggpubr)
library(caret)
library(randomForest)
library(xgboost)
library(Matrix)
library(MASS)
library(factoextra)
library(gbm)
library(glmnet)
library(plotmo)
set.seed(3)


# read data
train_dat <- read.csv("W22P1_train.csv", header = TRUE)
test_dat <- read.csv("W22P1_test.csv", header = TRUE)

test_dat = mutate(test_dat, trip_duration = rep(0, nrow(test_dat)), .before = 1)	
full_dat = rbind(train_dat, test_dat)

## Plot trip duration
ggplot(train_dat, aes(x=scale(trip_duration))) +
    geom_histogram(fill = "cornsilk",aes(y = ..density..))+
    geom_density() +
    xlim(0,1) +
    labs(x="scaled trip duration")

ggplot(train_dat, aes(x=log(trip_duration))) +
    geom_histogram(fill = "cornsilk",aes(y = ..density..))+
    geom_density() +
    labs(x="log trip duration")
    


# extract day from pickup_date
full_dat$day = day(full_dat$pickup_date)
# extract hour and mintue from pickup_time
full_dat$hour = hour(hms(full_dat$pickup_time))
full_dat$minute = minute(hms(full_dat$pickup_time))
full_dat$minuteofday = full_dat$hour * 60 + full_dat$minute
# Day of week, Sun == 1, Sat == 7
full_dat$weekday = wday(as.Date(full_dat$pickup_date))
# Weekend?
full_dat$weekend =  (full_dat$weekday == 1 | full_dat$weekday == 7) + 0

## plot date
full_dat[1:30000,] %>% 
    ggplot(aes(x=factor(weekday), y = log(trip_duration),fill=factor(weekday))) +
    geom_boxplot(outlier.colour="brown",  outlier.size=1, notch=FALSE)+
    labs(x='weekday', y='log_trip_duration')
full_dat[1:30000,] %>% 
    ggplot(aes(x=factor(weekend), y = log(trip_duration))) +
    geom_boxplot(outlier.colour="brown",  outlier.size=1, notch=FALSE)+
    labs(x='Weekend OR NOT', y='log_trip_duration')

full_dat[1:30000,] %>% 
    ggplot(aes(x=factor(hour), y = log(trip_duration),fill=factor(hour))) +
    geom_boxplot(outlier.colour="brown",  outlier.size=1, notch=FALSE)+
    labs(x='hour', y='log_trip_duration')
full_dat[1:30000,] %>% 
    ggplot(aes(x=factor(minute),y = log(trip_duration),fill=factor(minute))) +
    geom_boxplot(outlier.colour="brown",  outlier.size=1, notch=FALSE)+
    labs(x='minute', y='log_trip_duration')


# compute distance using Haversine
full_dat$dist_hav <- distHaversine(full_dat[,c("pickup_longitude","pickup_latitude")],
                                   full_dat[,c("dropoff_longitude","dropoff_latitude")])

# Haversine distance
geo_dist <- function(lon1, lat1, lon2, lat2){
    lon1 = lon1 * pi / 180
    lat1 = lat1 * pi / 180
    lon2 = lon2 * pi / 180
    lat2 = lat2 * pi / 180
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (sin(dlat/2))^2 + cos(lat1) * cos(lat2) * (sin(dlon/2))^2
    c = 2 * atan2( sqrt(a), sqrt(1-a) )
    return(6378137 * c)  # distance in meters
}
# distHaversine(full_dat[1,c("pickup_longitude","pickup_latitude")],
#               full_dat[1,c("dropoff_longitude","dropoff_latitude")])
# geo_dist(full_dat[1,"pickup_longitude"],full_dat[1,"pickup_latitude"],
#          full_dat[1,"dropoff_longitude"],full_dat[1,"dropoff_latitude"])


# driving direction
full_dat$direction <- bearing(full_dat[,c("pickup_longitude","pickup_latitude")],
                              full_dat[,c("dropoff_longitude","dropoff_latitude")])

# cluster on pickup location
WSS <- rep(0, 10)
for (k in 1:10) {
    # extract the total within-group sum of squared errors
    WSS[k] = kmeans(x = scale(full_dat[, 5:6]), centers = k, nstart = 100)$tot.withinss
}

ggplot(mapping = aes(x = 1:10, y = WSS)) +
    geom_line() +
    geom_point() +
    geom_vline(xintercept = 5) +
    scale_x_discrete(name = "k", limits = factor(1:10)) +
    labs(title = "Elbow Method")
# k = 3 or k = 5 is the best
pickup_kmeans <- kmeans(x = scale(full_dat[, 5:6]), centers = 5, nstart = 500)
full_dat$pickup_loc  = pickup_kmeans$cluster


# # cluster on dropoff location
WSS <- rep(0, 10)
for (k in 1:10) {
    # extract the total within-group sum of squared errors
    WSS[k] = kmeans(x = scale(full_dat[, 7:8]), centers = k, nstart = 100)$tot.withinss
}

ggplot(mapping = aes(x = 1:10, y = WSS)) +
    geom_line() +
    geom_point() +
    geom_vline(xintercept = 4) +
    scale_x_discrete(name = "k", limits = factor(1:10)) +
    labs(title = "Elbow Method")
# k = 3 or k = 4 is the best
dropoff_kmeans <- kmeans(x = scale(full_dat[, 7:8]), centers = 4, nstart = 500)
full_dat$dropoff_loc  = dropoff_kmeans$cluster



# plot kmeans on location
fviz_cluster(pickup_kmeans, data =full_dat[5:6], geom = "point")
fviz_cluster(dropoff_kmeans, data =full_dat[7:8], geom = "point")
# plot distance & duration
ggplot(full_dat[0:30000,],mapping = aes(x=scale(dist_hav), y=log(trip_duration)))+
    geom_point(color="brown",alpha = 0.5) + 
    geom_smooth(se=FALSE, color="orange")+
    xlim(0,15) +
    ylim(5,10) +
    labs(x="Haversine distance", y="log trip duration")

ggplot(full_dat[0:30000,],mapping = aes(x=direction, y=log(trip_duration)))+
    geom_point(color="Purple",alpha = 0.5) + 
    geom_smooth(se=FALSE)+
    labs(x="direction", y="log trip duration")


# standardize numeric cols: dist_hav, trip_duration
# columns useful after cleaning
#full_dat$dist_hav = scale(full_dat$dist_hav)
#full_dat$passenger_count = factor(full_dat$passenger_count)

# separate train_dat into train and validation group
# train, validationfull_dat = full_dat[, !(colnames(full_dat) %in% col)]
#col = c("log_trip_duration", "passenger_count", "day", "hour", "minuteofday",
#        "dist_hav", "pickup_loc", "dropoff_loc", "weekend", "direction")
col = c("id", "pickup_date","pickup_time", "trip_duration",
        "minuteofday", "pickup_loc", "dropoff_loc","weekday")

full_dat = full_dat[, !(colnames(full_dat) %in% col)]

full_dat = scale(full_dat)
full_dat$log_trip_duration = log(full_dat$log_trip_duration)

idx = sample(1:nrow(train_dat), round(nrow(train_dat)*3/4))

train_dat = full_dat[1:30000,]
train1 = train_dat[idx,]
validation1 = train_dat[-idx,]
# test
test1 = full_dat[30001:40000,]

box = boxplot(train1$log_trip_duration)$stats
train1 = filter(train1, log_trip_duration > box[1], log_trip_duration < box[5])

