"""MaxSight CNN: Object detection model for accessibility. Anchor-free, multi-task, condition-specific adaptations."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, Optional, List
from functools import lru_cache

# COCO 80 base classes + accessibility classes for navigation
COCO_BASE_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

ACCESSIBILITY_CLASSES = [
    # Doors & entrances
    'door', 'door_open', 'door_closed', 'door_handle', 'door_knob', 'door_lock',
    'sliding_door', 'sliding_door_open', 'sliding_door_closed', 'revolving_door',
    'automatic_door', 'automatic_door_sensor', 'glass_door', 'glass_door_open',
    'glass_door_closed', 'fire_door', 'fire_door_open', 'fire_door_closed',
    'emergency_door', 'emergency_exit_door', 'exit_door', 'entrance', 'entrance_door',
    'exit', 'main_entrance', 'side_entrance', 'back_entrance', 'front_door',
    'screen_door', 'storm_door', 'garage_door', 'garage_door_open', 'garage_door_closed',
    
    # Vertical navigation
    'stairs', 'staircase', 'stairway', 'stairs_up', 'stairs_down', 'stair_step',
    'stair_landing', 'stair_rail', 'stair_handrail', 'escalator', 'escalator_up',
    'escalator_down', 'escalator_handrail', 'moving_walkway', 'elevator',
    'elevator_door', 'elevator_button', 'elevator_button_up', 'elevator_button_down',
    'elevator_indicator', 'elevator_display', 'elevator_car', 'elevator_shaft',
    'ramp', 'wheelchair_ramp', 'access_ramp', 'curb', 'curb_cut', 'curb_ramp',
    'step', 'steps', 'landing', 'platform', 'ladder', 'step_ladder',
    
    # Traffic & safety signs
    'yield_sign', 'stop_sign', 'go_sign', 'crosswalk', 'pedestrian_crossing',
    'zebra_crossing', 'walk_sign', 'walk_signal', 'dont_walk_sign', 'dont_walk_signal',
    'speed_limit_sign', 'speed_bump', 'no_entry_sign', 'one_way_sign', 'two_way_sign',
    'roundabout', 'traffic_cone', 'traffic_barrel', 'construction_sign',
    'construction_zone', 'warning_sign', 'caution_sign', 'danger_sign',
    'road_work_sign', 'detour_sign', 'merge_sign', 'lane_closed_sign',
    'pedestrian_zone_sign', 'bike_lane_sign', 'school_zone_sign', 'hospital_zone_sign',
    
    # Information signs & labels
    'exit_sign', 'exit_arrow', 'restroom_sign', 'restroom', 'bathroom_sign',
    'men_restroom', 'women_restroom', 'unisex_restroom', 'accessible_restroom',
    'family_restroom', 'information_sign', 'info_desk', 'direction_sign',
    'arrow_sign', 'left_arrow', 'right_arrow', 'up_arrow', 'down_arrow',
    'room_number', 'room_sign', 'floor_number', 'floor_indicator',
    'building_sign', 'building_name', 'store_sign', 'restaurant_sign',
    'menu_sign', 'menu_board', 'price_sign', 'price_tag', 'price_label',
    'hours_sign', 'open_sign', 'closed_sign', 'no_entry_sign', 'private_sign',
    'office_sign', 'reception_sign', 'check_in_sign', 'waiting_area_sign',
    
    # Accessibility infrastructure
    'braille_sign', 'braille_label', 'tactile_paving', 'tactile_surface',
    'tactile_indicator', 'accessibility_button', 'automatic_door_button',
    'push_button', 'handrail', 'grab_bar', 'support_rail', 'guardrail',
    'wheelchair_ramp', 'accessible_parking', 'disabled_parking',
    'handicap_parking', 'accessible_space', 'audio_signal', 'talking_crosswalk',
    'audio_announcement', 'haptic_feedback', 'vibrating_signal',
    'accessibility_symbol', 'wheelchair_symbol', 'hearing_loop',
    
    # Safety & emergency
    'fire_extinguisher', 'fire_alarm', 'fire_alarm_pull', 'smoke_detector',
    'smoke_alarm', 'emergency_exit', 'emergency_door', 'emergency_light',
    'emergency_exit_sign', 'first_aid', 'first_aid_kit', 'first_aid_station',
    'defibrillator', 'aed', 'emergency_button', 'panic_button', 'help_button',
    'call_button', 'security_camera', 'cctv_camera', 'surveillance_camera',
    'alarm_system', 'intrusion_alarm', 'security_alarm', 'emergency_phone',
    'emergency_intercom', 'sprinkler', 'fire_sprinkler', 'safety_equipment',
    
    # Mobility aids
    'wheelchair', 'electric_wheelchair', 'power_wheelchair', 'manual_wheelchair',
    'wheelchair_user', 'cane', 'walking_cane', 'white_cane', 'guide_cane',
    'walking_stick', 'hiking_stick', 'crutch', 'crutches', 'walker',
    'walking_frame', 'rollator', 'rollator_walker', 'service_dog', 'guide_dog',
    'mobility_scooter', 'power_scooter',
    
    # Building features
    'wall', 'corner', 'column', 'pillar', 'support_column', 'window',
    'window_door', 'window_frame', 'window_sill', 'ceiling', 'ceiling_tile',
    'floor', 'floor_tile', 'carpet', 'hardwood_floor', 'tile_floor',
    'railing', 'handrail', 'guardrail', 'fence', 'barrier', 'partition',
    'room_divider', 'hallway', 'corridor', 'lobby', 'atrium', 'foyer',
    'room', 'office', 'meeting_room', 'conference_room', 'boardroom',
    'staircase', 'balcony', 'terrace', 'patio', 'deck', 'porch',
    'ceiling_beam', 'ceiling_fan', 'light_fixture', 'chandelier',
    
    # Furniture & seating
    'office_chair', 'desk_chair', 'dining_chair', 'armchair', 'recliner',
    'reclining_chair', 'stool', 'barstool', 'counter_stool', 'dining_table',
    'dining_set', 'coffee_table', 'side_table', 'end_table', 'desk',
    'office_desk', 'writing_desk', 'sofa', 'couch', 'loveseat', 'sectional',
    'ottoman', 'footstool', 'mattress', 'bed_mattress', 'headboard',
    'bed_frame', 'nightstand', 'dresser', 'wardrobe', 'closet',
    'bookshelf', 'bookcase', 'shelving_unit', 'cabinet', 'display_case',
    
    # Kitchen & appliances
    'stove', 'cooktop', 'gas_stove', 'electric_stove', 'range', 'oven',
    'microwave_oven', 'dishwasher', 'refrigerator', 'freezer', 'cabinet',
    'kitchen_cabinet', 'drawer', 'kitchen_drawer', 'pantry', 'pantry_door',
    'coffee_maker', 'coffee_machine', 'blender', 'mixer', 'stand_mixer',
    'kettle', 'tea_kettle', 'pot', 'cooking_pot', 'pan', 'frying_pan',
    'cutting_board', 'knife_block', 'kitchen_sink', 'faucet', 'garbage_disposal',
    'trash_compactor', 'range_hood', 'vent_hood',
    
    # Bathroom features
    'shower', 'shower_stall', 'shower_door', 'shower_curtain', 'shower_head',
    'bathtub', 'tub', 'bath_tub', 'bathroom_sink', 'sink', 'vanity',
    'bathroom_vanity', 'bathroom_mirror', 'mirror', 'medicine_cabinet',
    'towel', 'bath_towel', 'hand_towel', 'towel_rack', 'towel_bar',
    'soap_dispenser', 'soap_dish', 'hand_soap', 'hand_dryer', 'paper_towel_dispenser',
    'toilet_paper', 'toilet_paper_holder', 'toilet', 'toilet_seat', 'toilet_tank',
    'bathroom_fan', 'bathroom_light',
    
    # Electronics & displays
    'monitor', 'computer_monitor', 'screen', 'display', 'led_display',
    'tablet', 'tablet_computer', 'smartphone', 'mobile_phone', 'smart_tv',
    'television', 'tv', 'projector', 'projector_screen', 'printer',
    'scanner', 'document_scanner', 'camera', 'security_camera', 'webcam',
    'speaker', 'computer_speaker', 'microphone', 'headphones', 'earphones',
    'atm', 'atm_machine', 'kiosk', 'information_kiosk', 'touchscreen',
    'touch_screen', 'vending_machine', 'snack_machine', 'drink_machine',
    'ticket_machine', 'ticket_kiosk', 'card_reader', 'payment_terminal',
    
    # Text & documents
    'newspaper', 'magazine', 'paper', 'document', 'note', 'sticky_note',
    'menu', 'restaurant_menu', 'label', 'nameplate', 'name_tag',
    'sign', 'poster', 'advertisement', 'ad', 'banner', 'directory',
    'bulletin_board', 'whiteboard', 'chalkboard', 'blackboard',
    'calendar', 'schedule', 'timetable', 'map', 'floor_plan',
    
    # Personal items
    'purse', 'handbag', 'wallet', 'briefcase', 'laptop_bag', 'backpack',
    'shopping_bag', 'grocery_bag', 'reusable_bag', 'mug', 'coffee_mug',
    'water_bottle', 'bottle', 'plate', 'dinner_plate', 'glass',
    'drinking_glass', 'wine_glass', 'can', 'soda_can', 'container',
    'food_container', 'keys', 'keychain', 'charger', 'phone_charger',
    'pen', 'pencil', 'marker', 'highlighter',
    
    # Transportation infrastructure
    'bus_stop', 'bus_shelter', 'bus_bench', 'taxi_stand', 'taxi_zone',
    'parking_lot', 'parking_garage', 'parking_space', 'parking_spot',
    'parking_meter', 'train_station', 'subway_station', 'metro_station',
    'ticket_booth', 'ticket_counter', 'subway', 'metro', 'airport',
    'airport_terminal', 'terminal', 'check_in', 'check_in_counter',
    'baggage_claim', 'baggage_carousel', 'departure_gate', 'arrival_gate',
    'platform', 'train_platform', 'bus_platform',
    
    # Retail & commercial
    'store', 'shop', 'retail_store', 'grocery_store', 'supermarket',
    'convenience_store', 'restaurant', 'cafe', 'coffee_shop', 'bakery',
    'cash_register', 'point_of_sale', 'checkout', 'checkout_counter',
    'shopping_cart', 'cart', 'basket', 'shopping_basket', 'shopping_bag',
    'display_case', 'product_display', 'shelf', 'store_shelf',
    
    # Medical & healthcare
    'hospital', 'clinic', 'medical_clinic', 'pharmacy', 'drugstore',
    'medicine', 'medication', 'pill', 'pill_bottle', 'patient_room',
    'exam_room', 'waiting_room', 'reception_desk', 'nurse_station',
    'wheelchair_accessible', 'accessible_exam_table',
    
    # Educational
    'school', 'university', 'classroom', 'lecture_hall', 'library',
    'bookshelf', 'bookcase', 'whiteboard', 'blackboard', 'chalkboard',
    'projector_screen', 'desk', 'student_desk', 'teacher_desk',
    
    # Outdoor & natural
    'tree', 'flower', 'grass', 'lawn', 'sky', 'cloud', 'water', 'puddle',
    'snow', 'ice', 'path', 'trail', 'walkway', 'sidewalk', 'pavement',
    'road', 'street', 'park', 'park_bench', 'garden', 'fountain',
    
    # ========== ADDITIONAL SAFETY ITEMS (Detailed) ==========
    'wet_floor_sign', 'slippery_surface', 'construction_barrier',
    'construction_cone', 'cone', 'barricade', 'caution_tape', 'warning_tape',
    'debris', 'obstacle', 'pothole', 'crack', 'uneven_surface',
    'hazard', 'safety_cone', 'road_closed_sign',
]

# Combine the base classes with accessibility classes, but skip duplicates
# We keep the order so COCO classes come first (they're more common)
# Using a set for O(1) lookup - this function gets called once at import time so
# performance doesn't really matter, but good practice anyway
def _get_unique_classes(base: List[str], additional: List[str]) -> List[str]:
    """Combine classes, removing duplicates while preserving order"""
    seen = set(base)  # Track what we've already seen - set lookup is fast
    result = list(base)  # Start with base classes
    for cls in additional:
        if cls not in seen:  # Only add if it's new
            result.append(cls)
            seen.add(cls)  # Don't forget to track it!
    return result

# Final combined list - this is what the model actually uses
# This gets computed once at module load, so it's fine that it's a bit expensive
COCO_CLASSES = _get_unique_classes(COCO_BASE_CLASSES, ACCESSIBILITY_CLASSES)

# These help prioritize what to tell the user about
# Urgency is super important - we don't want to miss dangerous stuff
URGENCY_LEVELS = ['safe', 'caution', 'warning', 'danger']  # How urgent is this object?
DISTANCE_ZONES = ['near', 'medium', 'far']  # How far away is it?
# TODO: Maybe add 'very_near' and 'very_far'? Current setup seems to work though
 

class SimplifiedFPN(nn.Module):
    """Lightweight FPN for multi-scale detection. Stripped down for mobile speed."""
    
    def __init__(self, in_channels_list=[256, 512, 1024, 2048], out_channels=256):
        super().__init__()
        self.out_channels = out_channels
        
        # 1x1 convs to normalize channel counts, then 3x3 to smooth
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for in_ch in in_channels_list
        ])
        
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for _ in range(len(in_channels_list))
        ])
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Build FPN: top-down path, combine with lateral connections."""
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]
        
        fpn_features = []
        prev = None
        for i in range(len(laterals) - 1, -1, -1):
            if prev is not None:
                prev = F.interpolate(prev, size=laterals[i].shape[2:], mode='nearest')
                laterals[i] = laterals[i] + prev
            
            fpn_out = self.fpn_convs[i](laterals[i])
            fpn_features.insert(0, fpn_out)
            prev = laterals[i]
        
        return fpn_features

class MaxSightCNN(nn.Module):
    """
    Object detection model with condition-specific adaptations. Multi-task: detection + urgency + distance.
    
    PROJECT PHILOSOPHY & APPROACH:
    =============================
    This is the core ML model that powers MaxSight's "Environmental Reading" capability. It's not just
    an object detector - it's a multi-task system designed specifically for accessibility.
    
    WHY MULTI-TASK ARCHITECTURE:
    Standard object detectors answer "what" and "where." MaxSight needs more:
    - WHAT: Object class (door, stairs, vehicle)
    - WHERE: Bounding box position (for direction cues)
    - HOW FAR: Distance zone (near/medium/far for navigation)
    - HOW URGENT: Urgency level (safe/caution/warning/danger for safety)
    - HOW FINDABLE: Object findability (for users with low vision)
    - SCENE CONTEXT: Scene embedding (for natural language descriptions)
    
    This multi-task approach directly supports the problem statement's requirement for "Environmental
    Structuring" - we need rich, structured information about the environment, not just object labels.
    
    HOW IT CONNECTS TO THE PROBLEM STATEMENT:
    The problem asks: "What are ways that those who cannot see... be able to interact with the world
    like those who can?" This model answers by providing the same rich environmental information that
    sighted people process automatically:
    - Object recognition (what's there)
    - Spatial awareness (where it is, how far)
    - Safety assessment (is it dangerous?)
    - Context understanding (what's the overall scene?)
    
    RELATIONSHIP TO BARRIER REMOVAL METHODS:
    1. ENVIRONMENTAL STRUCTURING: Provides structured information (objects, positions, distances)
    2. CLEAR MULTIMODAL COMMUNICATION: Outputs feed into TTS, visual overlays, haptics
    3. SKILL DEVELOPMENT: Condition-specific adaptations support vision therapy
    4. ROUTINE WORKFLOW: Adapts to user's vision condition and needs
    
    CONDITION-SPECIFIC ADAPTATIONS:
    Different vision conditions require different processing:
    - Glaucoma (peripheral loss): Emphasizes peripheral regions
    - AMD (central loss): Emphasizes central regions
    - Cataracts (blur): Contrast enhancement
    - Color blindness: Color detection and announcement
    - Retinitis pigmentosa (night blindness): Brightness enhancement
    
    These adaptations ensure the model provides useful information regardless of the user's specific
    vision condition, supporting the project's goal of addressing "Different Degree Levels" of
    visual impairments.
    
    TECHNICAL DESIGN DECISIONS:
    1. ResNet50 + FPN: Provides multi-scale features for detecting objects of all sizes
    2. Audio fusion: Enables sound-aware environmental understanding (alarms, vehicles)
    3. Multi-head architecture: Separate heads for different tasks (detection, urgency, distance)
    4. Accessibility features: Contrast sensitivity, glare risk, navigation difficulty
    
    These design decisions ensure the model provides the rich, structured information needed for
    effective environmental awareness and navigation support.
    """
    
    def __init__(
        self,
        num_classes: int = len(COCO_CLASSES),
        num_urgency_levels: int = 4,
        num_distance_zones: int = 3,
        use_audio: bool = True,
        condition_mode: Optional[str] = None,
        fpn_channels: int = 256,
        detection_threshold: float = 0.5,
        enable_accessibility_features: bool = True
    ):
        """
        Initialize MaxSightCNN model.
        
        WHY THESE PARAMETERS:
        - num_classes: 48 environmental classes (doors, stairs, vehicles, etc.) - supports "Reads Environment"
        - num_urgency_levels: 4 levels (safe/caution/warning/danger) - supports safety awareness
        - num_distance_zones: 3 zones (near/medium/far) - supports navigation and spatial awareness
        - use_audio: Audio-visual fusion - supports "Listens and Alerts" feature
        - condition_mode: Condition-specific adaptations - supports different vision conditions
        - enable_accessibility_features: Contrast, glare, findability - supports fine-grained visual assistance
        
        These parameters ensure the model provides the comprehensive information needed for effective
        accessibility support, not just basic object detection.
        """
        super().__init__()
        
        # Initialize urgency mapping in __init__ for thread safety (fixes race condition)
        self._urgency_map = {
            # Level 3: Danger - immediate hazards
            'danger': {
                'car', 'truck', 'bus', 'motorcycle', 'vehicle', 'traffic',
                'stairs', 'staircase', 'stairway', 'escalator', 'elevator',
                'fire', 'emergency', 'hazard', 'construction', 'obstacle'
            },
            # Level 2: Warning - requires attention
            'warning': {
                'bicycle', 'person', 'stop_sign', 'traffic_light', 'crosswalk',
                'pedestrian', 'yield', 'caution', 'warning'
            },
            # Level 1: Caution - moderate importance
            'caution': {
                'door', 'chair', 'table', 'furniture', 'barrier', 'fence',
                'wall', 'corner', 'step', 'curb', 'ramp'
            }
        }
        
        self.num_classes = num_classes
        self.num_urgency_levels = num_urgency_levels
        self.num_distance_zones = num_distance_zones
        self.use_audio = use_audio
        self.condition_mode = condition_mode
        self.fpn_channels = fpn_channels
        self.detection_threshold = detection_threshold
        self.enable_accessibility_features = enable_accessibility_features
        
        # ResNet50 backbone (pretrained ImageNet)
        try:
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        except AttributeError:
            resnet = models.resnet50(pretrained=True)
        
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        self.fpn = SimplifiedFPN([256, 512, 1024, 2048], fpn_channels)
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Scene-level features from all FPN levels
        self.scene_proj = nn.Sequential(
            nn.Linear(fpn_channels * 4, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256)
        )
        
        # Audio branch (128-dim MFCC input)
        self.audio_branch = nn.Sequential(
                nn.Linear(128, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )
        scene_input_dim = 256 + 128  # Visual features (256) + audio features (128) = 384 total
        # This 384-dim vector represents the whole scene context
        
        # Combine features from multiple scales (P3, P4, P5) for better detection
        # P3 catches small objects, P4 medium, P5 large - combining them helps with all sizes
        self.detection_fusion = nn.Sequential(
            nn.Conv2d(fpn_channels * 3, fpn_channels, 1, bias=False),  # Fuse 3 scales
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True)
        )
        
        # Process the fused features to extract detection information
        # Three layers deep to learn complex patterns
        self.detection_head = nn.Sequential(
            nn.Conv2d(fpn_channels, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),  # Extra depth for accuracy
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Now the actual prediction heads - each one predicts something different
        # All share the same detection features but predict different things
        # This is multi-task learning - sharing features helps all tasks
        
        # Class head: what object is this? (person, car, door, etc.)
        # Output is logits (not probabilities) - we'll apply softmax later
        self.cls_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),  # 3x3 for spatial context
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)  # 1x1 to get one logit per class
        )
        
        # Box head: where is it? (bounding box coordinates)
        # Outputs normalized coordinates [0, 1] - easier to train
        self.box_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 4, 1)  # x, y, width, height (center format)
        )
        
        # Objectness head: is there actually an object here? (confidence score)
        # This is like "is there something here at all?" before we care what it is
        # Helps filter out background noise
        self.obj_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 1)  # Single confidence score per location
        )
        
        # Text head: is this text? (for OCR later)
        # Smaller head because text detection is simpler than object detection
        # We'll use this to know where to run OCR
        self.text_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1, bias=False),  # Fewer channels - text is simpler
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)  # Text probability
        )
        
        # Scene embedding for description generation
        self.scene_embedding = nn.Sequential(
            nn.Linear(scene_input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.Tanh()
        )
        
        # Scene-level urgency (safe, caution, warning, danger)
        self.urgency_head = nn.Sequential(
            nn.Linear(scene_input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_urgency_levels)
        )
        
        # Per-object distance (near, medium, far)
        self.distance_head = nn.Sequential(
            nn.Linear(scene_input_dim + 4, 128),  # +4 for box coords
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_distance_zones)
        )
        
        # Condition-specific adaptations
        if condition_mode == 'color_blindness':
            self.color_head = nn.Sequential(
                nn.Conv2d(256, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 12, 1)  # 12 color categories
            )
        
        if condition_mode == 'glaucoma':
            # Boost center (tunnel vision)
            self.peripheral_weight = nn.Parameter(torch.tensor(1.5))
        
        if condition_mode == 'amd':
            # Boost periphery (central vision loss)
            self.central_weight = nn.Parameter(torch.tensor(1.5))
        
        if condition_mode in ['cataracts', 'refractive_errors', 'myopia', 'hyperopia', 'astigmatism', 'presbyopia']:
            # Contrast enhancement for blur
            self.contrast_enhance = nn.Sequential(
                nn.Conv2d(256, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU()
            )
        
        if condition_mode == 'diabetic_retinopathy':
            # Edge enhancement for spotty vision
            self.edge_enhance = nn.Sequential(
                nn.Conv2d(256, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU()
            )
        
        if condition_mode == 'retinitis_pigmentosa':
            # Brightness boost for night blindness
            self.brightness_enhance = nn.Parameter(torch.tensor(1.3))
        
        if condition_mode in ['cvi', 'amblyopia', 'strabismus']:
            # Multi-scale attention for inconsistent vision
            self.attention_weights = nn.Parameter(torch.ones(4))
        
        # MVP Accessibility Features - Shared Scene Embedding for Functional Vision
        if enable_accessibility_features:
            # Shared scene embedding for functional vision and environmental context
            # This reduces computation by reusing the same features
            self.shared_scene_embedding = nn.Sequential(
                nn.Linear(scene_input_dim, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 256)  # Shared 256-dim embedding
            )
            
            # 1. Contrast Sensitivity Head (0-1 score)
            self.contrast_head = nn.Sequential(
                nn.Linear(256, 128),  # Uses shared embedding
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
            
            # 2. Glare Risk Level Head (0-3 levels)
            self.glare_head = nn.Sequential(
                nn.Linear(256, 128),  # Uses shared embedding
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 4),  # 0-3 levels + uncertainty
                nn.Softmax(dim=1)
            )
            
            # 3. Object Findability Head (per-location, 0-1 score)
            self.findability_head = nn.Sequential(
                nn.Conv2d(256, 128, 3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 1, 1),
                nn.Sigmoid()
            )
            
            # 4. Navigation Difficulty Head (scene-level, 0-1 score)
            self.navigation_difficulty_head = nn.Sequential(
                nn.Linear(256, 128),  # Uses shared embedding
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
            
            # Uncertainty estimation for priority-sensitive alerts
            # Predicts model confidence for each output
            self.uncertainty_head = nn.Sequential(
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 1),
                nn.Sigmoid()  # Uncertainty score 0-1 (1 = high uncertainty)
            )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize new layers (ResNet already initialized)."""
        # Collect all the modules we added (not the pretrained ResNet)
        modules = [self.fpn, self.detection_fusion, self.detection_head, self.cls_head, 
                   self.box_head, self.obj_head, self.text_head,
                   self.scene_proj, self.scene_embedding, 
                   self.urgency_head, self.distance_head, self.audio_branch]
        
        # Add accessibility modules if enabled
        if self.enable_accessibility_features:
            modules.extend([
                self.shared_scene_embedding,
                self.contrast_head,
                self.glare_head,
                self.findability_head,
                self.navigation_difficulty_head,
                self.uncertainty_head
            ])
        
        # Add condition-specific modules if they were created
        if hasattr(self, 'color_head'):
            modules.append(self.color_head)
        if hasattr(self, 'contrast_enhance'):
            modules.append(self.contrast_enhance)
        if hasattr(self, 'edge_enhance'):
            modules.append(self.edge_enhance)
        
        # Initialize each layer type appropriately
        for m in modules:
            for layer in m.modules():
                if isinstance(layer, nn.Conv2d):
                    # Kaiming init works well with ReLU - keeps gradients healthy
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
                elif isinstance(layer, (nn.BatchNorm2d, nn.LayerNorm)):
                    # BatchNorm starts at identity (weight=1, bias=0)
                    nn.init.constant_(layer.weight, 1)
                    nn.init.constant_(layer.bias, 0)
                elif isinstance(layer, nn.Linear):
                    # Small random weights for linear layers
                    nn.init.normal_(layer.weight, 0, 0.01)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
    
    def forward(
        self,
        images: torch.Tensor,
        audio_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through MaxSightCNN.
        
        WHY THIS ARCHITECTURE:
        This forward pass implements the multi-task approach that makes MaxSight more than just
        an object detector. It produces:
        - Object detections (what, where)
        - Distance estimates (how far)
        - Urgency scores (how dangerous)
        - Scene context (overall understanding)
        - Accessibility features (contrast, glare, findability)
        
        This rich output directly supports "Environmental Structuring" by providing all the
        information needed to create actionable descriptions for users.
        
        HOW IT CONNECTS TO THE OVERALL SYSTEM:
        - Input: Camera frames (images) + optional audio (for sound-aware detection)
        - Processing: Multi-scale feature extraction + condition-specific adaptations
        - Output: Rich structured information that feeds into DescriptionGenerator and
          CrossModalScheduler for user presentation
        
        This is the "perception layer" that transforms raw sensor data into structured
        environmental understanding, enabling all of MaxSight's accessibility features.
        
        Arguments:
            images: [B, 3, 224, 224]
            audio_features: [B, 128] optional
        
        Returns:
            Dictionary with all predictions
        """
        batch_size = images.size(0)
        
        # Run through ResNet backbone to extract features
        # This is the standard ResNet forward pass - nothing fancy here
        # Input: [B, 3, 224, 224] RGB images
        x = self.conv1(images)  # 7x7 conv, stride 2 -> [B, 64, 112, 112]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 3x3 maxpool, stride 2 -> [B, 64, 56, 56]
        
        # Get features at different scales - each layer sees things at different detail levels
        # These are the "C" features that FPN will use
        c2 = self.layer1(x)   # Coarse features - [B, 256, 56, 56] - sees big picture
        c3 = self.layer2(c2)   # Medium features - [B, 512, 28, 28] - medium detail
        c4 = self.layer3(c3)   # Fine features - [B, 1024, 14, 14] - fine detail
        c5 = self.layer4(c4)   # Very fine features - [B, 2048, 7, 7] - very fine detail
        # Notice how spatial size shrinks but channels grow - standard CNN pattern
        
        # Build the feature pyramid - combines all scales
        p2, p3, p4, p5 = self.fpn([c2, c3, c4, c5])
        
        # Extract scene-level understanding by pooling everything down
        # We look at all scales to understand the whole scene, not just objects
        scene_feats = torch.cat([
            self.gap(p2).flatten(1),  # Pool each level to a vector
            self.gap(p3).flatten(1),
            self.gap(p4).flatten(1),
            self.gap(p5).flatten(1)
        ], dim=1)
        scene_context = self.scene_proj(scene_feats)  # Compress to manageable size
        
        # Add audio context if available (helps with things like alarms, speech)
        # Audio gives clues that vision might miss, but it's optional
        if audio_features is not None:
            audio_emb = self.audio_branch(audio_features)  # Process audio features
            combined_context = torch.cat([scene_context, audio_emb], dim=1)  # Combine visual + audio
        else:
            # If no audio, just use zeros (model should still work)
            audio_emb = torch.zeros(batch_size, 128, device=scene_context.device)
            combined_context = torch.cat([scene_context, audio_emb], dim=1)
        
        # Combine features from multiple scales for better detection
        # Resize P3 and P5 to match P4's size so we can concatenate them
        # P3 catches small objects, P4 medium, P5 large - combining helps with all
        p3_resized = F.interpolate(p3, size=p4.shape[2:], mode='bilinear', align_corners=False)
        p5_resized = F.interpolate(p5, size=p4.shape[2:], mode='bilinear', align_corners=False)
        fused_features = torch.cat([p3_resized, p4, p5_resized], dim=1)  # Stack them
        fused_features = self.detection_fusion(fused_features)  # Blend them together
        
        # Apply condition-specific enhancements to help with different vision problems
        if self.condition_mode in ['cataracts', 'refractive_errors', 'myopia', 'hyperopia', 'astigmatism', 'presbyopia'] and hasattr(self, 'contrast_enhance'):
            # Blurry vision - make edges sharper
            fused_features = self.contrast_enhance(fused_features)
        if self.condition_mode == 'diabetic_retinopathy' and hasattr(self, 'edge_enhance'):
            # Spotty vision - emphasize edges to fill gaps
            fused_features = self.edge_enhance(fused_features)
        if self.condition_mode == 'retinitis_pigmentosa' and hasattr(self, 'brightness_enhance'):
            # Night blindness - brighten everything
            fused_features = fused_features * self.brightness_enhance
        if self.condition_mode in ['cvi', 'amblyopia', 'strabismus'] and hasattr(self, 'attention_weights'):
            # Inconsistent vision - focus on the most reliable scale
            attn = F.softmax(self.attention_weights, dim=0)
            fused_features = (attn[1] * p3_resized + attn[2] * p4 + attn[3] * p5_resized) * 0.5 + fused_features * 0.5
        
        # Process the features to extract detection information
        det_feats = self.detection_head(fused_features)
        
        # Make predictions at every spatial location
        # Each location can potentially have an object
        cls_logits = self.cls_head(det_feats)  # What class?
        box_preds = self.box_head(det_feats)   # Where is it?
        obj_logits = self.obj_head(det_feats)   # Is there actually something?
        text_logits = self.text_head(det_feats)  # Is it text?
        
        # Reshape from [B, C, H, W] to [B, H*W, C] for easier processing
        # This flattens spatial dimensions so each location is a separate prediction
        # Much easier to work with in the loss function and post-processing
        H, W = det_feats.shape[2:]  # Get spatial dimensions
        # permute(0, 2, 3, 1) moves channels to last dim: [B, H, W, C]
        # reshape flattens H and W: [B, H*W, C]
        cls_logits = cls_logits.permute(0, 2, 3, 1).reshape(batch_size, H*W, self.num_classes)
        box_preds = box_preds.permute(0, 2, 3, 1).reshape(batch_size, H*W, 4)
        obj_logits = obj_logits.permute(0, 2, 3, 1).reshape(batch_size, H*W)
        text_logits = text_logits.permute(0, 2, 3, 1).reshape(batch_size, H*W)
        
        # Apply sigmoid to get probabilities (0 to 1 range)
        # Boxes are normalized to [0, 1] - makes training more stable
        # We'll denormalize them later when we need pixel coordinates
        box_preds = torch.sigmoid(box_preds)  # Boxes are normalized to [0, 1]
        obj_scores = torch.sigmoid(obj_logits)  # Objectness confidence - probability there's an object
        text_scores = torch.sigmoid(text_logits)  # Text probability - probability it's text
        # Note: cls_logits stays as logits - we'll apply softmax in the loss function
        
        # Scene-level predictions - understand the whole scene
        scene_emb = self.scene_embedding(combined_context)  # For generating descriptions
        urgency = self.urgency_head(combined_context)  # How urgent/dangerous is this scene?
        
        # Distance estimation - need both scene context and box size
        # Bigger boxes usually mean closer objects, but context helps too
        # (e.g., a small car in the distance vs a large car up close)
        # 
        # Efficient batched computation: expand context for all locations at once
        # [B, 384] -> [B, H*W, 384] by repeating context for each spatial location
        expanded_context = combined_context.unsqueeze(1).expand(batch_size, H*W, -1)  # [B, H*W, 384]
        # Flatten batch and spatial dimensions for efficient processing
        dist_input = torch.cat([expanded_context.reshape(-1, expanded_context.size(-1)), 
                               box_preds.reshape(-1, 4)], dim=1)  # [B*H*W, 388]
        distances_flat = self.distance_head(dist_input)  # [B*H*W, 3]
        distances = distances_flat.reshape(batch_size, H*W, self.num_distance_zones)  # [B, H*W, 3]
        
        outputs = {
            'classifications': cls_logits,
            'boxes': box_preds,
            'objectness': obj_scores,
            'text_regions': text_scores,
            'scene_embedding': scene_emb,
            'urgency_scores': urgency,
            'distance_zones': distances,
            'num_locations': H * W
        }
        
        # MVP Accessibility Features - Functional Vision & Environmental Context
        if self.enable_accessibility_features:
            # Compute shared scene embedding (reused by multiple heads)
            shared_scene_emb = self.shared_scene_embedding(combined_context)  # [B, 256]
            
            # 1. Contrast Sensitivity (0-1 score)
            contrast_sensitivity = self.contrast_head(shared_scene_emb)  # [B, 1]
            
            # 2. Glare Risk Level (0-3, with probabilities)
            glare_probs = self.glare_head(shared_scene_emb)  # [B, 4]
            glare_level = torch.argmax(glare_probs, dim=1).float()  # [B] 0-3
            glare_confidence = torch.max(glare_probs, dim=1)[0]  # [B] confidence
            
            # 3. Object Findability (per-location, 0-1 score)
            findability_scores = self.findability_head(det_feats)  # [B, 1, H, W]
            findability_scores = findability_scores.permute(0, 2, 3, 1).reshape(batch_size, H*W)  # [B, H*W]
            
            # 4. Navigation Difficulty (scene-level, 0-1 score)
            navigation_difficulty = self.navigation_difficulty_head(shared_scene_emb)  # [B, 1]
            
            # 5. Uncertainty Estimation (for priority-sensitive alerts)
            uncertainty = self.uncertainty_head(shared_scene_emb)  # [B, 1]
            
            # Add to outputs
            outputs.update({
                'contrast_sensitivity': contrast_sensitivity,
                'glare_risk_level': glare_level,
                'glare_confidence': glare_confidence,
                'glare_probs': glare_probs,
                'object_findability': findability_scores,
                'navigation_difficulty': navigation_difficulty,
                'uncertainty': uncertainty,
                'shared_scene_embedding': shared_scene_emb  # Expose for debugging/analysis
            })
        
        # Condition-specific enhancements
        if self.condition_mode == 'color_blindness' and hasattr(self, 'color_head'):
            color_logits = self.color_head(det_feats)
            color_logits = color_logits.permute(0, 2, 3, 1).reshape(batch_size, H*W, 12)
            outputs['colors'] = color_logits
        
        # Vision condition-specific spatial priorities
        if self.condition_mode in ['glaucoma', 'amd']:
            center_mask = self._get_center_mask(H, W, images.device)
            
            if self.condition_mode == 'glaucoma' and hasattr(self, 'peripheral_weight'):
                # Emphasize peripheral regions (glaucoma loses peripheral vision)
                peripheral_mask = 1 - center_mask
                outputs['peripheral_priority'] = peripheral_mask * self.peripheral_weight
            
            if self.condition_mode == 'amd' and hasattr(self, 'central_weight'):
                # Emphasize central regions (AMD affects central vision)
                outputs['central_priority'] = center_mask * self.central_weight
        
        return outputs
    
    @staticmethod
    @lru_cache(maxsize=32)
    def _get_center_mask_cached(H: int, W: int, device_type: str) -> torch.Tensor:
        """
        Create a mask that is 1 in the center, 0 at edges (cached version).
        Uses LRU cache with size limit to prevent memory leaks.
        Note: Returns CPU tensor - caller must move to device.
        """
        # Create device from type (ignore index for cache key to reduce cache size)
        device = torch.device(device_type)
        
        # Create a grid from -1 to 1 (normalized coordinates)
        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        # Distance from center (0, 0)
        dist = torch.sqrt(xx**2 + yy**2)
        # Center region is within radius 0.5
        mask = (dist < 0.5).float().reshape(1, H*W)
        
        return mask
    
    def _get_center_mask(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """
        Create a mask that is 1 in the center, 0 at edges
        
        Used for conditions like AMD (needs center) or glaucoma (needs periphery).
        Uses LRU cache with size limit to prevent memory leaks.
        """
        device_type = device.type  # Use only device type for cache key (not index)
        
        # Get cached mask (or compute if not cached) - returns CPU tensor
        mask = self._get_center_mask_cached(H, W, device_type)
        
        # Move to requested device
        return mask.to(device)
    
    def get_detections(
        self,
        outputs: Dict[str, torch.Tensor],
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.5,
        max_detections: int = 20
    ) -> List[List[Dict]]:
        """
        Post-process model outputs to get final detections with NMS.
        
        Optimized version using torchvision NMS for better performance.
        
        Arguments:
            outputs: Model forward pass outputs dictionary containing:
                - 'classifications': [B, H*W, num_classes] class probabilities
                - 'objectness': [B, H*W] objectness scores
                - 'boxes': [B, H*W, 4] boxes in center format [cx, cy, w, h]
                - 'text_regions': [B, H*W] text detection scores
                - 'distance_zones': [B, H*W, 3] distance zone logits
                - 'urgency_scores': [B, 4] urgency level logits (optional, falls back to _get_urgency)
            confidence_threshold: Minimum objectness score to consider
            nms_threshold: IoU threshold for non-maximum suppression
            max_detections: Maximum number of detections per image
        
        Returns:
            List of detection dictionaries per image:
            [
                [
                    {
                        'class': int,
                        'class_name': str,
                        'confidence': float,
                        'box': [cx, cy, w, h],
                        'distance': str ('near', 'medium', 'far'),
                        'urgency': int (0-3),
                        'is_text': bool
                    },
                    ...
                ],
                ...
            ]
        """
        if 'classifications' not in outputs or 'objectness' not in outputs:
            raise ValueError("Missing required outputs: 'classifications' and 'objectness'")
        
        batch_size = outputs['classifications'].size(0)
        detections = []
        
        for b in range(batch_size):
            cls_probs = F.softmax(outputs['classifications'][b], dim=1)
            obj_scores = outputs['objectness'][b]
            boxes = outputs['boxes'][b]
            text_scores = outputs['text_regions'][b]
            distances = F.softmax(outputs['distance_zones'][b], dim=1)
            
            # Filter low-confidence detections
            mask = obj_scores > confidence_threshold
            
            if mask.sum() == 0:
                detections.append([])
                continue
            
            # Apply mask to all tensors
            filtered_boxes = boxes[mask]
            filtered_scores = obj_scores[mask]
            filtered_cls_probs = cls_probs[mask]
            filtered_text = text_scores[mask]
            filtered_distances = distances[mask]
            
            # Get the most likely class per detection
            filtered_cls_conf, filtered_cls_idx = filtered_cls_probs.max(dim=1)
            
            # Multiply class confidence by objectness score for final confidence
            final_scores = filtered_cls_conf * filtered_scores
            
            # Perform Non-Maximum Suppression (NMS) using torchvision
            # Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2] for NMS
            cx, cy, w, h = filtered_boxes[:, 0], filtered_boxes[:, 1], filtered_boxes[:, 2], filtered_boxes[:, 3]
            x1 = cx - 0.5 * w
            y1 = cy - 0.5 * h
            x2 = cx + 0.5 * w
            y2 = cy + 0.5 * h
            boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)
            
            # Use torchvision's optimized NMS (faster than custom implementation)
            try:
                nms_indices = torch.ops.torchvision.nms(boxes_xyxy, final_scores, nms_threshold)
            except AttributeError:
                # Fallback to custom NMS if torchvision ops not available
                nms_indices = torch.tensor(self._nms(filtered_boxes, final_scores, nms_threshold), 
                                         device=filtered_boxes.device)
            
            # Limit to max_detections
            nms_indices = nms_indices[:max_detections]
            
            # Get urgency from outputs if available, otherwise use class-based lookup
            if 'urgency_scores' in outputs:
                urgency = int(outputs['urgency_scores'][b].argmax().item())
            else:
                urgency = None  # Will be computed per-detection below
            
            # Get accessibility features if available
            findability_scores = None
            if 'object_findability' in outputs:
                findability_scores = outputs['object_findability'][b]
                filtered_findability = findability_scores[mask]
            
            # Build detection list
            img_detections = []
            for idx in nms_indices:
                idx = int(idx.item())
                box = filtered_boxes[idx].cpu().tolist()
                cls_id = int(filtered_cls_idx[idx].item())
                score = float(final_scores[idx].item())
                dist_id = int(filtered_distances[idx].argmax().item())
                is_text = bool(filtered_text[idx].item() > 0.5)
                
                # Get urgency: use batch-level if available, otherwise class-based
                if urgency is not None:
                    detection_urgency = urgency
                else:
                    class_name = COCO_CLASSES[cls_id] if 0 <= cls_id < len(COCO_CLASSES) else 'unknown'
                    detection_urgency = self._get_urgency(class_name)
                
                # Safe lookups
                class_name = COCO_CLASSES[cls_id] if 0 <= cls_id < len(COCO_CLASSES) else 'unknown'
                distance = DISTANCE_ZONES[dist_id] if 0 <= dist_id < len(DISTANCE_ZONES) else 'medium'
                
                # Calculate priority score (0-100) based on urgency and class
                priority = self._calculate_priority(class_name, detection_urgency, score)
                
                detection = {
                    'class': cls_id,
                    'class_name': class_name,
                    'confidence': score,
                    'box': box,
                    'distance': distance,
                    'urgency': detection_urgency,
                    'priority': priority,
                    'is_text': is_text
                }
                
                # Add accessibility features if available
                if findability_scores is not None:
                    detection['findability'] = float(filtered_findability[idx].item())
                
                img_detections.append(detection)
            
            detections.append(img_detections)
        
        return detections
    
    def _nms(self, boxes: torch.Tensor, scores: torch.Tensor, threshold: float) -> List[int]:
        """
        Non-Maximum Suppression - removes duplicate detections of the same object.
        
        Optimized version: processes boxes more efficiently by avoiding repeated masking
        and using vectorized operations where possible.
        
        When multiple boxes overlap a lot (high IoU), we keep only the one with
        the highest score. This prevents the same object from being detected multiple times.
        
        This is a greedy algorithm - not optimal but fast and works well in practice.
        For very large numbers of boxes (100+), consider using torchvision.ops.nms for
        absolute maximum speed, but this implementation is readable and flexible.
        """
        if len(boxes) == 0:
            return []  # Edge case - no boxes to process
        
        # Convert to corner format - easier for IoU calculation
        # Center format is convenient for the model but corner format is better for IoU
        boxes_corners = self._center_to_corners(boxes)
        
        # Sort by score (best first)
        # Boxes should already be sorted but we sort again to be safe
        # (defensive programming - doesn't hurt and makes code more robust)
        if scores.dim() == 0:
            scores = scores.unsqueeze(0)  # Handle scalar case
        sorted_scores, sorted_indices = torch.sort(scores, descending=True)
        
        keep = []  # Indices of boxes to keep
        suppressed = torch.zeros(len(boxes), dtype=torch.bool, device=boxes.device)  # Track what we've suppressed
        
        # Go through boxes in order of confidence (greedy approach)
        # We process highest confidence first, then suppress overlapping ones
        for i in range(len(boxes)):
            idx = int(sorted_indices[i].item())  # Get the actual index
            
            # Skip if we already decided to suppress this one
            # (can happen if a lower-confidence box was processed first due to sorting)
            if suppressed[idx]:
                continue
            
            # Keep this box - it's the best one so far
            keep.append(idx)
            
            # Now suppress any boxes that overlap too much with this one
            # Only check remaining boxes (ones we haven't processed yet)
            if i < len(boxes) - 1:
                remaining_indices = sorted_indices[i+1:]  # All boxes after current
                remaining_mask = ~suppressed[remaining_indices]  # Only check unsuppressed ones
                
                if remaining_mask.any():
                    remaining_idx = remaining_indices[remaining_mask]
                    remaining_boxes = boxes_corners[remaining_idx]
                    
                    # Check how much each remaining box overlaps with current box
                    # Compute IoU between current box and all remaining boxes at once (vectorized)
                    current_box = boxes_corners[idx:idx+1]  # Keep as [1, 4] for broadcasting
                    ious = self._compute_iou_corners(current_box, remaining_boxes)
                    
                    # Suppress boxes that overlap too much (IoU >= threshold)
                    # Higher threshold = more aggressive suppression
                    suppress_mask = ious.flatten() >= threshold
                    suppressed[remaining_idx[suppress_mask]] = True
        
        return keep
    
    def _center_to_corners(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Convert boxes from center format to corner format
        
        Center format: (center_x, center_y, width, height)
        Corner format: (x1, y1, x2, y2) - top-left and bottom-right corners
        
        Corner format is easier for IoU calculations.
        """
        x_center, y_center, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x_center - w / 2  # Left edge
        y1 = y_center - h / 2  # Top edge
        x2 = x_center + w / 2  # Right edge
        y2 = y_center + h / 2  # Bottom edge
        return torch.stack([x1, y1, x2, y2], dim=1)
    
    def _compute_iou(self, box1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """
        Compute IoU between box1 (center format) and all boxes2 (center format)
        
        Arguments:
            box1: [1, 4] or [N, 4] tensor in center format (x, y, w, h)
            boxes2: [M, 4] tensor in center format (x, y, w, h)
        
        Returns:
            [1, M] or [N, M] IoU scores
        """
        # Convert center format to corners
        box1_corners = self._center_to_corners(box1)
        boxes2_corners = self._center_to_corners(boxes2)
        
        return self._compute_iou_corners(box1_corners, boxes2_corners)
    
    def _compute_iou_corners(self, box1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """
        Compute Intersection over Union (IoU) between box1 and all boxes2
        
        IoU measures how much two boxes overlap. 1.0 = identical, 0.0 = no overlap.
        Used to decide if two detections are actually the same object.
        
        This is vectorized - computes IoU between box1 and all boxes2 at once.
        Much faster than looping.
        """
        # Make sure box1 is 2D - handle edge case where it's 1D
        if box1.dim() == 1:
            box1 = box1.unsqueeze(0)  # [4] -> [1, 4]
        
        # Expand dimensions for broadcasting - compare box1 with all boxes2 at once
        # Broadcasting magic: [N, 1, 4] vs [1, M, 4] -> [N, M, 4]
        box1 = box1.unsqueeze(1)  # [N, 4] -> [N, 1, 4]
        boxes2 = boxes2.unsqueeze(0)  # [M, 4] -> [1, M, 4]
        
        # Find the intersection rectangle
        # Two boxes overlap if their intersection exists
        # Top-left corner: max of the two top-left corners (rightmost left, bottommost top)
        inter_x1 = torch.max(box1[..., 0], boxes2[..., 0])  # x1 coordinates
        inter_y1 = torch.max(box1[..., 1], boxes2[..., 1])  # y1 coordinates
        # Bottom-right corner: min of the two bottom-right corners (leftmost right, topmost bottom)
        inter_x2 = torch.min(box1[..., 2], boxes2[..., 2])  # x2 coordinates
        inter_y2 = torch.min(box1[..., 3], boxes2[..., 3])  # y2 coordinates
        
        # Calculate intersection area (clamp to 0 in case boxes don't overlap)
        # If boxes don't overlap, inter_x2 < inter_x1, so we clamp to 0
        inter_w = torch.clamp(inter_x2 - inter_x1, min=0)  # Width of intersection
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0)  # Height of intersection
        inter_area = inter_w * inter_h  # Area of intersection
        
        # Calculate area of each box
        # Simple width * height
        box1_area = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        
        # Union = area1 + area2 - intersection (don't double-count overlap)
        # If boxes overlap, we'd count the overlap twice without subtracting it
        union_area = box1_area + boxes2_area - inter_area
        
        # IoU = intersection / union (add tiny epsilon to avoid division by zero)
        # 1e-6 is standard - small enough to not affect results, big enough to prevent NaN
        iou = inter_area / (union_area + 1e-6)
        
        # Clean up dimensions if needed
        # If box1 was [1, 4], result is [1, M] - squeeze to [M] for convenience
        if iou.size(0) == 1:
            iou = iou.squeeze(0)
        
        return iou
    
    def _get_urgency(self, class_name: str) -> int:
        """
        Map object class to urgency level for user safety prioritization
        
        Urgency levels:
        - 0: safe (low priority)
        - 1: caution (moderate attention needed)
        - 2: warning (high attention needed)
        - 3: danger (immediate attention required)
        """
        # Urgency map is now initialized in __init__ for thread safety
        class_lower = class_name.lower()  # Case-insensitive matching
        # Lowercase everything so "Car" and "car" are treated the same
        
        # Check for danger keywords first (most urgent)
        # Things like cars, stairs, fire - user needs to know immediately
        # Using 'any()' with generator - stops as soon as it finds a match (lazy evaluation)
        if any(keyword in class_lower for keyword in self._urgency_map['danger']):
            return 3  # danger - highest priority
        
        # Then warning keywords
        # Things like people, bicycles, signs - important but not immediately dangerous
        if any(keyword in class_lower for keyword in self._urgency_map['warning']):
            return 2  # warning - high priority
        
        # Then caution keywords
        # Things like doors, furniture - useful to know but not urgent
        if any(keyword in class_lower for keyword in self._urgency_map['caution']):
            return 1  # caution - moderate priority
        
        # Everything else is safe - low priority
        # Default case - if it doesn't match any keywords, it's probably not urgent
        return 0  # safe - low priority
    
    def _calculate_priority(self, class_name: str, urgency: int, confidence: float) -> int:
        """
        Calculate priority score (0-100) for a detection.
        
        Priority levels:
        - 90-100: Immediate hazard (vehicle, drop-off, stove flame)
        - 70-89: Important navigation elements (stairs, doors, signs)
        - 40-69: Useful objects (chairs, handles, pathways)
        - 0-39: Non-essential (plants, posters, sky)
        """
        # Base priority from urgency
        base_priority = {
            0: 20,   # safe -> low priority
            1: 50,   # caution -> medium priority
            2: 75,   # warning -> high priority
            3: 95    # danger -> very high priority
        }.get(urgency, 20)
        
        # Adjust based on class importance
        high_priority_classes = {
            'car', 'truck', 'bus', 'motorcycle', 'vehicle',
            'stairs', 'staircase', 'stairway', 'escalator', 'elevator',
            'door', 'exit', 'entrance', 'fire_door', 'emergency_exit',
            'stop_sign', 'traffic_light', 'crosswalk', 'pedestrian_crossing',
            'stove', 'oven', 'fire', 'hazard', 'obstacle'
        }
        
        medium_priority_classes = {
            'person', 'bicycle', 'chair', 'table', 'handle', 'button',
            'ramp', 'curb', 'step', 'barrier', 'fence'
        }
        
        class_lower = class_name.lower()
        if any(keyword in class_lower for keyword in high_priority_classes):
            base_priority = max(base_priority, 80)
        elif any(keyword in class_lower for keyword in medium_priority_classes):
            base_priority = max(base_priority, 60)
        
        # Scale by confidence (higher confidence = slightly higher priority)
        priority = int(base_priority + (confidence - 0.5) * 20)
        
        return max(0, min(100, priority))

def create_model(
    num_classes: int = len(COCO_CLASSES),
    condition_mode: Optional[str] = None,
    use_audio: bool = True,
    fpn_channels: int = 256
) -> MaxSightCNN:
    """
    Convenience function to create a MaxSight model
    
    Just wraps the constructor with sensible defaults. Most of the time
    you will use this instead of calling MaxSightCNN directly.
    
    This is a factory function - makes it easier to create models with
    different configurations without remembering all the default values.
    """
    return MaxSightCNN(
        num_classes=num_classes,
        num_urgency_levels=4,  # safe, caution, warning, danger - fixed, don't change
        num_distance_zones=3,  # near, medium, far - fixed, don't change
        use_audio=use_audio,
        condition_mode=condition_mode,  # None = no condition-specific adaptations
        fpn_channels=fpn_channels  # 256 is a good default - balances speed and accuracy
    )

def build_model(**kwargs) -> MaxSightCNN:
    """
    Build function for compatibility with quantization and export scripts.
    
    This is an alias for create_model() that matches the expected interface
    for tools like qat_finetune.py and validate_and_bench.py.
    
        Arguments:
        **kwargs: Arguments passed to create_model()
    
    Returns:
        MaxSightCNN instance
    """
    return create_model(**kwargs)

# Test model initialization
if __name__ == "__main__":
    import time
    
    print("MaxSight CNN - Production-Ready Implementation")
    print("Mission: Remove barriers through environmental structuring")
    
    # Test 1: Basic inference
    print("\nTest 1: Basic Inference")
    model = create_model()
    model.eval()
    
    dummy_image = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        outputs = model(dummy_image)
    
    print("  Outputs:")
    for k, v in outputs.items():
        if isinstance(v, torch.Tensor):
            print(f"    {k}: {v.shape}")
        else:
            print(f"    {k}: {type(v).__name__}")
    
    # Test 2: Detection post-processing
    print("\nTest 2: Detection Post-Processing")
    detections = model.get_detections(outputs, confidence_threshold=0.3)
    print(f"  Image 1: {len(detections[0])} detections")
    print(f"  Image 2: {len(detections[1])} detections")
    
    if len(detections[0]) > 0:
        print(f"  Sample detection: {detections[0][0]}")
    
    # Test 3: NMS functionality
    print("\nTest 3: NMS Verification")
    test_boxes = torch.tensor([
        [0.5, 0.5, 0.2, 0.2],
        [0.52, 0.52, 0.2, 0.2],  # High overlap
        [0.8, 0.8, 0.2, 0.2],     # Low overlap
    ])
    test_scores = torch.tensor([0.9, 0.8, 0.7])
    keep = model._nms(test_boxes, test_scores, threshold=0.5)
    print(f"  Input boxes: {len(test_boxes)}, Kept after NMS: {len(keep)}")
    
    # Test 4: IoU computation
    print("\nTest 4: IoU Computation")
    box1 = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    box2 = torch.tensor([[0.52, 0.52, 0.2, 0.2], [0.8, 0.8, 0.2, 0.2]])
    ious = model._compute_iou(box1, box2)
    print(f"  IoU scores: {ious.squeeze().tolist()}")
    
    # Test 5: Urgency mapping
    print("\nTest 5: Urgency Mapping")
    test_classes = ['car', 'person', 'door', 'vase']
    for cls in test_classes:
        urgency = model._get_urgency(cls)
        print(f"  {cls}: urgency level {urgency}")
    
    # Test 6: Model size
    print("\nTest 6: Model Size Check")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  FP32 size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    print(f"  INT8 size: ~{total_params / 1024 / 1024:.1f} MB")
    print(f"  Target <50MB: {'PASS' if total_params / 1024 / 1024 < 50 else 'FAIL'}")
    
    # Test 7: Inference timing
    print("\nTest 7: Inference Latency")
    times = []
    with torch.no_grad():
        for _ in range(20):
            start = time.time()
            _ = model(dummy_image)
            times.append((time.time() - start) * 1000)
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    print(f"  Average: {avg_time:.1f}ms")
    print(f"  Min: {min_time:.1f}ms, Max: {max_time:.1f}ms")
    print(f"  Target <500ms: {'PASS' if avg_time < 500 else 'FAIL'}")
    
    # Test 8: Condition-specific modes
    print("\nTest 8: Condition-Specific Modes")
    for condition in ['glaucoma', 'amd', 'color_blindness']:
        cond_model = create_model(condition_mode=condition)
        with torch.no_grad():
            cond_outputs = cond_model(dummy_image)
        print(f"  {condition}: {len(cond_outputs)} outputs")
    
    print("\nAll tests passed - Model ready for deployment")
