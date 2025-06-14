# tests/test_donations.py
import unittest
from app import app

class TestDonations(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_get_donations(self):
        response = self.app.get('/donations/')
        self.assertEqual(response.status_code, 200)

    def test_create_donation(self):
        response = self.app.post('/donations/', json={
            "food_type": "biryani",
            "quantity": "10 plates",
            "location": "Mumbai",
            "donor_phone": "+919876543210",
            "expiry_risk": "low"
        })
        self.assertEqual(response.status_code, 201)

if __name__ == '__main__':
    unittest.main()
