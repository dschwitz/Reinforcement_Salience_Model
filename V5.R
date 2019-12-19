library(dplyr)
library(ggplot2)

############### experiment parameters ############### 

n.subjects <- 60 # how many simulated subjects to include per experiment
n.trials <- 50 # how many simulated trials completed per subject
prob.reward.A <- .6 # the probability of a positive reward for chosing option A
prob.reward.B <- .4 # the probability of a positive reward for chosing option B
reward.value.A <- 1 # the value of reward A
reward.value.B <- 1 # the value of reward B
## Reward A or B may be negative (i.e., a "punishment," rather than a reward)
## and may have different values (e.g., -1 and 1). 
base.salience.reward.A <- abs(reward.value.A) # the base valence-independent salience
# of reward A (i.e., the intensity)
base.salience.reward.B <- abs(reward.value.B) # the base valence-independent salience
# of reward B (i.e., the intensity)
## In this model, we operationalize intensity as the base valence-independent
## salience of outcomes. Intensity hinges on how "strong" an outcome is. 
## Therefore, the base amount of VIS assigned to a given reward/punishment
## stimulus is equivalent to the absolute value of the hedonic value of the
## stimulus. For instance, if reward A has a value of 1, and reward B has a value of
## -1, both will have a base valence-independent base salience of 1. 
prob.arousing.factor.A <- .5 # the probability of the subject being exposed to
# an arousing factor given choice A
prob.arousing.factor.B <- .1 # the probability of the subject being exposed to
# an arousing factor given choice B
arousing.factor.intensity <- 2 # the intensity of the arousing factor
## Previous research has found that arousal impacts salience (Sutherland et al,
## 2018). In this model, we simulate the induction of "arousal" in subjects with a
## chance of an "arousing factor" with a certain intensity. 
novelty.bonus <- 1.5 # The salience-impacting "bonus" due to the first presentation
# of a stimulus 
## Previous research has found that novelty impacts salience (Foley et al, 2014).
## In this model, we operationalize "novelty" via a bonus depending on whether or
## not a stimulus has already been presented to a subject. This bonus directly 
## modifies the impact of salience, like the arousing factor. 

############### implementing softmax ############### 

## Here, the softmax algorithm is used to simulate each choice made for each
## simulated trial. Given an internally represented value of choice A and choice
## B, it outputs the probability of selecting either choice, normalized along a 
## 0-1 probabilistic scale. "Temperature" determines the explore/exploit tradeoff
## of the decision (i.e., the adjusted probability of choosing higher-valued
## options). Therefore, this function constitutes the policy of our model. 

softmax.rule <- function(a.value, b.value, temperature){
  return(
    exp(a.value*(1/temperature))/
      (exp(a.value*(1/temperature)) + exp(b.value*(1/temperature)))
  )
}

############### model ############### 

run.one.subject <- function(){
  
  alpha <- runif(1, .08, .12) # assigns an alpha value to each subject
  ## In this model, alpha represents the learning rate. Higher alphas result
  ## in faster associations between reward outcomes and policy. 
  temperature <- runif(1, .1, .15) # assigns a temperature value to each
  # subject (for the softmax algorithm)
  choices.history <- c() # records the history of choices made by each subject
  prob.a.history <- c() # records the probability of choosing A (determined by
  #the softmax function) in each trial
  trial.num <- 1
  a.value <- 0
  b.value <- 0
  choice <- 'a'
  salience <- 1
  
  for(i in c(trial.num:n.trials)) {
    
    ## Use the sofmax algorithm to make a choice:
    prob.a <- softmax.rule(a.value, b.value, temperature)
    choice <- (sample(c('a','b'), 1, prob=c(prob.a, 
                                            (1-prob.a))))
    choices.history <- c(choices.history, choice)
    
    prob.a.history <- c(prob.a.history, prob.a)
    
    ## If "A" was chosen...
    if(choice == 'a') {
      ## start at baseline salience
      salience <- 1
      ## Determine the reward/punishment received: 
      reward <- (sample(c(reward.value.A, 0), 1, prob=c(prob.reward.A, 
                                                        (1-prob.reward.A))))
      ## if the reward/punishment of choice A is received, assign salience:
      if (reward == reward.value.A) {
        salience <- salience*base.salience.reward.A
        ## additionally, if the stimulus has not yet been presented, assign the
        ## novelty bonus to salience:
        if(sum(choices.history=='a') == 1) {
          salience <- salience*novelty.bonus
        }
      }
    
      ## Determine whether or not the arousing factor is presented:
      if (sample(c(1, 0), 1, prob=c(prob.arousing.factor.A,
                                    1-prob.arousing.factor.A))) {
        ## if so, modify salience accordingly:
        salience <- salience*arousing.factor.intensity
      }
      
      ## Modify the internal representation of the value of A:
      a.value <- a.value + (salience*alpha)*(reward - a.value)
      ## This function simulates the learning process. The new value of choice A
      ## is equivalent to the previous value of A plus the reward prediction
      ## error (represented by the difference between the received reward and the
      ## previous value of A) multiplied the alpha (learning rate) times total
      ## salience. 
      
      ## If "B" was chosen...
    } else {
      salience <- 1
      ## Determine the reward/punishment received: 
      reward <- (sample(c(reward.value.B, 0), 1, prob=c(prob.reward.B, 
                                                        (1-prob.reward.B)))) 
      ## if the reward/punishment of choice B is received, assign salience:
      if (reward == reward.value.B) {
        salience <- salience*base.salience.reward.B
        ## additionally, if the stimulus has not yet been presented, assign the
        ## novelty bonus to salience:
        if(sum(choices.history=='b') == 1) {
          salience <- salience*novelty.bonus
        }
      }
      
      ## Determine whether or not the arousing factor is presented:
      if (sample(c(1, 0), 1, prob=c(prob.arousing.factor.B,
                                    1-prob.arousing.factor.B))) {
        ## if so, modify salience accordingly:
        salience <- salience*arousing.factor.intensity
      } 
      
      ## Modify the internal representation of the value of B:
      b.value <- b.value + (salience*alpha)*(reward - b.value)
      ## This function simulates the learning process as noted above, but for
      ## choice B.
    }
    
    
  }
  
  
  return(data.frame(trial=1:n.trials, choice=choices.history, probability=prob.a.history))
}

## Set up a data frame to hold experiment data: 

experiment.data <- NA 
for(s in 1:n.subjects){ 
  subject.data <- run.one.subject() 
  if(is.na(experiment.data)){ 
    experiment.data <- subject.data 
  } else { 
    experiment.data <- rbind(experiment.data, subject.data)
  }
}

## Simulate one experiment and record data:

run.one.experiment <- function() {
  experiment.data <- NA 
  for(s in 1:n.subjects){ 
    subject.data <- run.one.subject() 
    if(is.na(experiment.data)){ 
      experiment.data <- subject.data 
    } else { 
      experiment.data <- rbind(experiment.data, subject.data) 
    }
  }
  summarized.data <- experiment.data %>%
    group_by(trial) %>%
    summarize(choice_pct = sum(choice=="a")/n(), prob_mean = mean(probability))
  return(mean(summarized.data$choice_pct))
}

## Summarize and plot data (to produce a similar figure to Fig 2 in the paper)

summarized.data <- experiment.data %>%
  group_by(trial) %>%
  summarize(choice_pct = sum(choice=="a")/n(), prob_mean = mean(probability))

ggplot(summarized.data, aes(x=trial, y=choice_pct))+
  geom_point()+
  geom_line(aes(y=prob_mean))+
  ylim(c(0,1))+
  labs(x="Trial Number", y="% Chose A")+
  theme_bw()

print(mean(summarized.data$choice_pct))

## Run multiple experiments (is slightly more computationally efficient than running 
## thousands of participants in a single experiment)

run.n.experiments <- function(n) {
  results <- c()
  for(i in 1:n) {
    results <- c(results, run.one.experiment())
  }
  return(mean(results))
}

## Iterate over a certain range, and modify a parameter by a certain interval at each
## loop; finally, summarize and graph the results. This produces a similar figure to
## Fig. 3-6 in the paper. 

val = c()

result = c()

for(i in 1:9){
  print("loop")
  prob.arousing.factor.B <- prob.arousing.factor.B + .045
  val <- c(val, prob.arousing.factor.B)
  result <- c(result, run.n.experiments(150))
}

all.results <- data.frame(val = val, result = result)

ggplot(all.results, aes(x=val, y=result))+
  geom_point()+
  geom_line(aes(y=result))+
  ylim(c(.5, 1))+
  labs(x="Probability of AF for B", y="Mean % Chose A")+
  theme_bw()

## Acknowledgements: the bare-bones foundation of this model was built
## from a starter file provided by Prof. Joshua R. de Leeuw for his
## seminar on modeling in cognitive science. To access this starter
## file, contact him at jdeleeuw@vassar.edu
