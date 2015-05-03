

class Target(object):
    """
    Tracks a target, stores all measurements, provides model-dependent futures.

    (Purely a sketch/stub, currently)
    """
    active = False
    new_data = False

    futures = None

    def update_target_futures(tgt, event_models, future_epochs):
        for cls in event_models:
            tgt.class_fits[cls] = fit_class_likelihoods(tgt,cls)
            tgt.lightcurve_futures[cls] = project_lightcurve_futures(tgt,cls)
            tgt.confusion_per_epoch = bayes.calc_confusion_matrices(tgt.futures)