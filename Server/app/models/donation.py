
class Donation:
    def __init__(self, food_type, quantity, location, donor_phone, expiry_risk, matched_ngo=None):
        self.food_type = food_type
        self.quantity = quantity
        self.location = location
        self.donor_phone = donor_phone
        self.expiry_risk = expiry_risk
        self.matched_ngo = matched_ngo
