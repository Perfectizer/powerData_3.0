class Power:
    def __init__(self, user_id, stamp_time, power_value):
        self.user_id = user_id
        self.stamp_time = stamp_time
        self.power_value = power_value

    def get_power_value(self):
        return self.power_value
    def get_user_id(self):
        return self.user_id
    def get_stamp_time(self):
        return self.stamp_time
