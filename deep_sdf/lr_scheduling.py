class LearningRateSchedule:
    def get_learning_rate(self, epoch, *args, **kwargs):
        pass


class ConstantLearningRateSchedule(LearningRateSchedule):
    def __init__(self, value):
        self.value = value

    def get_learning_rate(self, epoch, *args, **kwargs):
        return self.value


class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch, *args, **kwargs):

        return self.initial * (self.factor ** (epoch // self.interval))
        

class StepLearningRateOnPlateauSchedule(LearningRateSchedule):
    def __init__(self, initial, factor, patience, threshold, min_lr, cooldown=0) -> None:
        """
        Reduce learning rate when the loss has stopped improving. 

        initial: initial LR value
        factor: factor by which the learning rate will be reduced
        patience: number of epochs with no improvement after which learning rate will be reduced
        threshold: threshold for measuring the new optimum, to only focus on significant changes (change value). 
            Say we have patience=100 and threshold=0.0001, if loss is 18.0 on epoch n and loss is 17.9999 on epoch n+1 
            then we have met our criteria to multiply the current learning rate by the factor.
        min_lr: minimum learning rate to use (cut-off value)
        cooldown: number of steps after the last learning rate update to not do another update
        """
        self.initial = initial
        self.min_lr = min_lr
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.last_lr = initial
        self.last_step_epoch = -99999       
        self.cooldown = cooldown

    def get_learning_rate(self, epoch, loss_log=[], *args, **kwargs):
        if len(loss_log) > self.patience and epoch - self.last_step_epoch > self.cooldown:
            before_patience_min = min(loss_log[:-self.patience])
            in_patience_min = min(loss_log[-self.patience:])
            if before_patience_min - in_patience_min <= self.threshold:
                self.last_lr *= self.factor
                self.last_step_epoch = epoch
        return max(self.last_lr, self.min_lr)


class WarmupLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, warmed_up, length):
        self.initial = initial
        self.warmed_up = warmed_up
        self.length = length

    def get_learning_rate(self, epoch, *args, **kwargs):
        if epoch > self.length:
            return self.warmed_up
        return self.initial + (self.warmed_up - self.initial) * epoch / self.length


def get_learning_rate_schedules(specs):

    schedule_specs = specs["LearningRateSchedule"]
    assert schedule_specs != [], "LearningRateSchedule NEEDS TO CONTAIN TWO DICTS BUT WAS EMPTY."

    schedules = []
    for schedule_specs in schedule_specs:

        if schedule_specs["Type"] == "Step":
            schedules.append(
                StepLearningRateSchedule(
                    schedule_specs["Initial"],
                    schedule_specs["Interval"],
                    schedule_specs["Factor"],
                )
            )
        elif schedule_specs["Type"] == "Warmup":
            schedules.append(
                WarmupLearningRateSchedule(
                    schedule_specs["Initial"],
                    schedule_specs["Final"],
                    schedule_specs["Length"],
                )
            )
        elif schedule_specs["Type"] == "Constant":
            schedules.append(ConstantLearningRateSchedule(schedule_specs["Value"]))
        elif schedule_specs["Type"] == "StepOnPlateau":
            schedules.append(StepLearningRateOnPlateauSchedule(
                schedule_specs["Initial"],
                schedule_specs["Factor"],
                schedule_specs["Patience"],
                schedule_specs["Threshold"],
                schedule_specs["MinLR"],
                schedule_specs.get("Cooldown"),
            ))
        else:
            raise Exception(f'no known learning rate schedule of type "{schedule_specs["Type"]}"')

    return schedules
