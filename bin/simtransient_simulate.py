
#### Skeleton outline for collecting ideas
from copy import copy

event_models_rates = { a: "",
                       b:"",
                       c:"",
                        }
# (These know about rates)


science_priorities = { a: 1, b:0.1, c:0.7, }

set_seed(123456)

all_sim_epochs = []
survey_epochs = all_sim_epochs[::4]

sim_events = generate_event_list( event_model_rates )

survey_data = {e.id : simulate_detections(e, survey_epochs)
               for e in sim_events }

scheduler = None # Greedy, call out to UCT, whatever

recommended_actions = {} #Recommended actions at each epoch

rescheduling_required=False

targets = {} # Map transient id to Target object

# Record targets, recommended actions, at each epoch
target_history = {}
recommendation_history = {}
action_history = {}

for idx, epoch in enumerate(all_sim_epochs):
    if idx<len(epochs)-1:
        next_epoch = epochs[idx+1]
    future_epochs = epochs[>epoch]

    if epoch in survey_epochs:
        for event_id in survey_data:
            if event_id not in targets:
                ### Test if event is blind-detected in this survey epoch
                source_obs = survey_data[event_id][epoch]
                if is_blind_detected(source_obs):
                    targets[event_id]= Target(source_obs, active=True)

        for tgt in targets:
            tgt.get_observation(epoch, survey_datasource)
            if tgt.active:
                tgt.update_futures(event_models, future_epochs)

    if recommended_actions[epoch]:
        for target, datasource in recommended_actions[epoch]:
            target.get_observation(datasource)

        action_history[epoch] = recommended_actions[epoch]

    recommended_actions = scheduler.make_recommendations(targets,
                                                     science_priorities)

    target_history[epoch] = copy(targets)
    recommendation_history[epoch] = copy(recommended_actions)




