# app/routes/donations.py
from flask import Blueprint, request, jsonify
from app.models.donation import Donation
from app.services import supabase_service
from app.services.twilio_service import send_sms

donations_bp = Blueprint('donations', __name__)

@donations_bp.route('/', methods=['GET'])
def get_donations():
    supabase = supabase_service.get_supabase()
    data = supabase.table('food_donations').select("*").execute()
    return jsonify(data.data), 200

@donations_bp.route('/', methods=['POST'])
def create_donation():
    supabase = supabase_service.get_supabase()
    data = request.json
    
    # Inserting donation data into Supabase
    response = supabase.table('food_donations').insert(data).execute()
    
    # Fetch NGOs based on location
    ngos_response = supabase.table('ngos').select("*").eq('location', data['location']).execute()
    ngos = ngos_response.data
    
    if ngos:
        # Pick the first NGO for now (can be randomized or prioritized later)
        ngo = ngos[0]
        ngo_phone_number = ngo['phone_number']
        
        # Sending SMS
        message = f"New food donation available for pickup!\nLocation: {data['location']}\nQuantity: {data['quantity']}"
        send_sms(ngo_phone_number, message)
        print(f"✅ SMS sent to NGO: {ngo['name']} at {ngo_phone_number}")
    else:
        print("⚠️ No NGOs found for this location.")

    return jsonify(response.data), 201



@donations_bp.route('/<donation_id>', methods=['PUT'])
def update_donation(donation_id):
    supabase = supabase_service.get_supabase()
    data = request.json
    response = supabase.table('food_donations').update(data).eq('id', donation_id).execute()
    return jsonify(response.data), 200

@donations_bp.route('/<donation_id>', methods=['DELETE'])
def delete_donation(donation_id):
    supabase = supabase_service.get_supabase()
    response = supabase.table('food_donations').delete().eq('id', donation_id).execute()
    return jsonify({"message": "Donation deleted successfully"}), 200
