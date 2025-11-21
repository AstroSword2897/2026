# This gathers the torch imports for the MaxSight CNN
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, Optional, List


# These are the different objects that are necessary for the CNN
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
    
    # Safety and Traffic
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
    
    # Building objects
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
    
    # Transportation
    'bus_stop', 'bus_shelter', 'bus_bench', 'taxi_stand', 'taxi_zone',
    'parking_lot', 'parking_garage', 'parking_space', 'parking_spot',
    'parking_meter', 'train_station', 'subway_station', 'metro_station',
    'ticket_booth', 'ticket_counter', 'subway', 'metro', 'airport',
    'airport_terminal', 'terminal', 'check_in', 'check_in_counter',
    'baggage_claim', 'baggage_carousel', 'departure_gate', 'arrival_gate',
    'platform', 'train_platform', 'bus_platform',
    
    # Shop
    'store', 'shop', 'retail_store', 'grocery_store', 'supermarket',
    'convenience_store', 'restaurant', 'cafe', 'coffee_shop', 'bakery',
    'cash_register', 'point_of_sale', 'checkout', 'checkout_counter',
    'shopping_cart', 'cart', 'basket', 'shopping_basket', 'shopping_bag',
    'display_case', 'product_display', 'shelf', 'store_shelf',
    
    # Medical
    'hospital', 'clinic', 'medical_clinic', 'pharmacy', 'drugstore',
    'medicine', 'medication', 'pill', 'pill_bottle', 'patient_room',
    'exam_room', 'waiting_room', 'reception_desk', 'nurse_station',
    'wheelchair_accessible', 'accessible_exam_table',
    
    # Educational
    'school', 'university', 'classroom', 'lecture_hall', 'library',
    'bookshelf', 'bookcase', 'whiteboard', 'blackboard', 'chalkboard',
    'projector_screen', 'desk', 'student_desk', 'teacher_desk',
    
    # Nature
    'tree', 'flower', 'grass', 'lawn', 'sky', 'cloud', 'water', 'puddle',
    'snow', 'ice', 'path', 'trail', 'walkway', 'sidewalk', 'pavement',
    'road', 'street', 'park', 'park_bench', 'garden', 'fountain',
    
    # Extra items
    'wet_floor_sign', 'slippery_surface', 'construction_barrier',
    'construction_cone', 'cone', 'barricade', 'caution_tape', 'warning_tape',
    'debris', 'obstacle', 'pothole', 'crack', 'uneven_surface',
    'hazard', 'safety_cone', 'road_closed_sign',
]

def _get_unique_classes(base: List[str], additional: List[str]) -> List[str]:
    seen = set(base)
    result = list(base)
    for cls in additional:
        if cls not in seen:
            result.append(cls)
            seen.add(cls)
    return result
COCO_CLASSES = _get_unique_classes(COCO_BASE_CLASSES, ACCESSIBILITY_CLASSES)

URGENCY_LEVELS = ['safe', 'caution', 'warning', 'danger']
DISTANCE_ZONES = ['near', 'medium', 'far']

# This class creates a lightweight FPN model used for the channels off of MaxSightCNN
class SimplifiedFPN(nn.Module):
    def __init__(self, in_channels_list=[256, 512, 1024, 2048], out_channels=256):
        super().__init__()
        self.out_channels = out_channels
        
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
    def __init__(
        self,
        num_classes: int = len(COCO_CLASSES),
        num_urgency_levels: int = 4,
        num_distance_zones: int = 3,
        use_audio: bool = True,
        condition_mode: Optional[str] = None,
        fpn_channels: int = 256,
        detection_threshold: float = 0.5,      
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_urgency_levels = num_urgency_levels
        self.num_distance_zones = num_distance_zones
        self.use_audio = use_audio
        self.condition_mode = condition_mode
        self.fpn_channels = fpn_channels
        self.detection_threshold = detection_threshold
        
        try:
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
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
        self.fpn = SimplifiedFPN([256, 512, 1024, 2048], out_channels=fpn_channels)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.scene_proj = nn.Sequential(
            nn.Linear(fpn_channels * 4, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256)
        )
        self.audio_branch = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )
        ## 256 from visual and 128 from audio
        scene_input_dim = 384

        self.detection_fusion = nn.Sequential(
            nn.Conv2d(fpn_channels * 3, fpn_channels, 1, bias=False), 
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True)
        )
        
        self.detection_head = nn.Sequential(
            nn.Conv2d(fpn_channels, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.cls_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self.box_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 4, 1)
        )
        self.obj_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 1)
        )
        self.text_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 1)
        )
        self.scene_embedding = nn.Sequential(
            nn.Linear(scene_input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512), 
            nn.Tanh()
        )
        self.urgency_head = nn.Sequential(
            nn.Linear(scene_input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2), 
            nn.Linear(256, num_urgency_levels)
        )

        self.distance_head = nn.Sequential(
            nn.Linear(scene_input_dim + 4, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_distance_zones)
        )
        
        # Condition-specific heads
        # Supports: refractive_errors, cataracts, glaucoma, amd, diabetic_retinopathy,
        #           retinitis_pigmentosa, color_blindness, cvi, amblyopia, strabismus
        if condition_mode == 'color_blindness':
            # Color classification head for color-blind users
            self.color_head = nn.Sequential(
                nn.Conv2d(256, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 12, 1)  # 12 color categories
            )
        
        if condition_mode == 'glaucoma':
            # Peripheral vision emphasis (glaucoma loses peripheral vision)
            self.peripheral_weight = nn.Parameter(torch.tensor(1.5))
        
        if condition_mode == 'amd':
            # Central vision emphasis (AMD affects central vision)
            self.central_weight = nn.Parameter(torch.tensor(1.5))
        
        if condition_mode in ['cataracts', 'refractive_errors', 'myopia', 'hyperopia', 'astigmatism', 'presbyopia']:
            # Enhanced contrast and sharpness for blurry vision (refractive errors + cataracts)
            # Handles: myopia (nearsighted), hyperopia (farsighted), astigmatism, presbyopia (aging lens)
            self.contrast_enhance = nn.Sequential(
                nn.Conv2d(256, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU()
            )
        
        if condition_mode == 'diabetic_retinopathy':
            # Enhanced edge detection for spotty/blurry vision
            self.edge_enhance = nn.Sequential(
                nn.Conv2d(256, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU()
            )
        
        if condition_mode == 'retinitis_pigmentosa':
            # Brightness enhancement for night blindness/tunnel vision
            self.brightness_enhance = nn.Parameter(torch.tensor(1.3))
        
        if condition_mode in ['cvi', 'amblyopia', 'strabismus']:
            #Higher level of attention for inconsistent vision
            self.attention_weights = nn.Parameter(torch.ones(4))  # For P2, P3, P4, P5
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize new layers"""
        modules = [self.fpn, self.detection_fusion, self.detection_head, self.cls_head, 
                   self.box_head, self.obj_head, self.text_head,
                   self.scene_proj, self.scene_embedding, 
                   self.urgency_head, self.distance_head, self.audio_branch]
        
        # Add condition-specific modules if they exist
        if hasattr(self, 'color_head'):
            modules.append(self.color_head)
        if hasattr(self, 'contrast_enhance'):
            modules.append(self.contrast_enhance)
        if hasattr(self, 'edge_enhance'):
            modules.append(self.edge_enhance)
        
        for m in modules:
            for layer in m.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
                elif isinstance(layer, (nn.BatchNorm2d, nn.LayerNorm)):
                    nn.init.constant_(layer.weight, 1)
                    nn.init.constant_(layer.bias, 0)
                elif isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, 0, 0.01)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
    
    def forward(
        self,
        images: torch.Tensor,
        audio_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass - Environmental reading
        
        Arguments:
            images: [B, 3, 224, 224]
            audio_features: [B, 128] optional
        
        Returns:
            Dictionary with all predictions
        """
        batch_size = images.size(0)
        
        # Backbone
        x = self.conv1(images)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        
        # FPN
        p2, p3, p4, p5 = self.fpn([c2, c3, c4, c5])
        
        # Scene features
        scene_feats = torch.cat([
            self.gap(p2).flatten(1),
            self.gap(p3).flatten(1),
            self.gap(p4).flatten(1),
            self.gap(p5).flatten(1)
        ], dim=1)
        scene_context = self.scene_proj(scene_feats)
        
        # Audio features: alarms, speech, environmental sounds (128-dim MFCC)
        if audio_features is not None:
            audio_emb = self.audio_branch(audio_features)  # [B, 128]
            combined_context = torch.cat([scene_context, audio_emb], dim=1)  # [B, 384]
        else:
            # Fallback: use zeros if audio not provided 
            audio_emb = torch.zeros(batch_size, 128, device=scene_context.device)
            combined_context = torch.cat([scene_context, audio_emb], dim=1)
        
        # Multi-scale detection fusion for high accuracy - combines P3, P4, P5
        p3_resized = F.interpolate(p3, size=p4.shape[2:], mode='bilinear', align_corners=False)
        p5_resized = F.interpolate(p5, size=p4.shape[2:], mode='bilinear', align_corners=False)
        fused_features = torch.cat([p3_resized, p4, p5_resized], dim=1) 
        fused_features = self.detection_fusion(fused_features)
        
        # Apply condition-specific visual enhancements as it is the main target area
        if self.condition_mode in ['cataracts', 'refractive_errors', 'myopia', 'hyperopia', 'astigmatism', 'presbyopia'] and hasattr(self, 'contrast_enhance'):
            # Enhance contrast for blurry vision (refractive errors + cataracts)
            fused_features = self.contrast_enhance(fused_features)  
        if self.condition_mode == 'diabetic_retinopathy' and hasattr(self, 'edge_enhance'):
            # Enhance edges for spotty vision
            fused_features = self.edge_enhance(fused_features)  
        if self.condition_mode == 'retinitis_pigmentosa' and hasattr(self, 'brightness_enhance'):
            fused_features = fused_features * self.brightness_enhance  
        if self.condition_mode in ['cvi', 'amblyopia', 'strabismus'] and hasattr(self, 'attention_weights'):
            # Attention for inconsistent vision
            attn = F.softmax(self.attention_weights, dim=0)
            fused_features = (attn[1] * p3_resized + attn[2] * p4 + attn[3] * p5_resized) * 0.5 + fused_features * 0.5
        
        det_feats = self.detection_head(fused_features)
        
        # Per-location predictions
        cls_logits = self.cls_head(det_feats)
        box_preds = self.box_head(det_feats)
        obj_logits = self.obj_head(det_feats)
        text_logits = self.text_head(det_feats)
        
        # Reshape
        H, W = det_feats.shape[2:]
        cls_logits = cls_logits.permute(0, 2, 3, 1).reshape(batch_size, H*W, self.num_classes)
        box_preds = box_preds.permute(0, 2, 3, 1).reshape(batch_size, H*W, 4)
        obj_logits = obj_logits.permute(0, 2, 3, 1).reshape(batch_size, H*W)
        text_logits = text_logits.permute(0, 2, 3, 1).reshape(batch_size, H*W)
        
        # Activations
        box_preds = torch.sigmoid(box_preds)
        obj_scores = torch.sigmoid(obj_logits)
        text_scores = torch.sigmoid(text_logits)
        
        # Scene-level
        scene_emb = self.scene_embedding(combined_context)
        urgency = self.urgency_head(combined_context)
        
        # Per-detection distance
        distances = []
        for i in range(batch_size):
            ctx = combined_context[i:i+1].expand(H*W, -1)
            boxes = box_preds[i]
            dist_input = torch.cat([ctx, boxes], dim=1)
            distances.append(self.distance_head(dist_input))
        distances = torch.stack(distances, dim=0)
        
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

        #Condition based enhancements
        if self.condition_mode == 'color_blindness' and hasattr(self, 'color_head'):
            color_logits = self.color_head(det_feats)
            color_logits = color_logits.permute(0,2,3,1).reshape(batch_size, H*W, 12)
            outputs['colors'] = color_logits

        #Spatial priority based off of visual issue
        if self.condition_mode in ['glaucoma', 'amd']:
            center_mask = self._get_center_mask(H,W,images.device)
            if self.condition_mode == 'glaucoma' and hasattr(self,'peripheral_weight'):
                peripheral_mask = 1 - center_mask
                outputs['peripheral_priority'] = peripheral_mask * self.peripheral_weight
            if self.condition_mode == 'amd' and hasattr(self,'central_weight'):
                outputs['central_priority'] = center_mask * self.central_weight
        return outputs
        
    def _get_center_mask(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Generate center region mask for spatial priority weighting"""
        cache_key = (H, W, str(device))
        if not hasattr(self, '_mask_cache'):
            self._mask_cache = {}
        
        if cache_key not in self._mask_cache:
            y = torch.linspace(-1, 1, H, device=device)
            x = torch.linspace(-1, 1, W, device=device)
            yy, xx = torch.meshgrid(y, x, indexing='ij')
            dist = torch.sqrt(xx**2 + yy**2)
            self._mask_cache[cache_key] = (dist < 0.5).float().reshape(1, H*W)
        
        return self._mask_cache[cache_key]
    
    def get_detections(
        self,
        outputs: Dict[str, torch.Tensor],
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.5,
        max_detections: int = 20
    ) -> List[Dict]:
        '''
        Arguments:
            outputs: Model forward pass outputs dictionary
            confidence_threshold: Minimum objectness score to consider
            nms_threshold: IoU threshold for non-maximum suppression
            max_detections: Maximum number of detections per image
        '''

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

            mask = obj_scores > confidence_threshold
            num_valid = mask.sum().item()
            
            if num_valid == 0:
                detections.append([])
                continue
            filtered_scores = obj_scores[mask]
            filtered_cls = cls_probs[mask]
            filtered_boxes = boxes[mask]
            filtered_text = text_scores[mask]
            filtered_distances = distances[mask]

            cls_conf, cls_idx = filtered_cls.max(dim=1)

            final_scores = filtered_scores * cls_conf

            sorted_idx = torch.argsort(final_scores, descending=True)
            keep_indices = self._nms(
                filtered_boxes[sorted_idx],
                final_scores[sorted_idx],
                nms_threshold
            )
            keep_indices = keep_indices[:max_detections]
            img_detections = []
            for idx in keep_indices:
                orig_idx = sorted_idx[idx]
                class_id = int(cls_idx[orig_idx].item())
                distance_zone = int(torch.argmax(filtered_distances[orig_idx]).item())
                class_name = COCO_CLASSES[class_id] if 0 <= class_id < len(COCO_CLASSES) else 'unknown'
                distance =  DISTANCE_ZONES[distance_zone] if 0 <= distance_zone < len(COCO_CLASSES) else 'medium'
                img_detections.append({
                    'class': class_id,
                    'class_name': class_name,
                    'confidence': float(final_scores[orig_idx].item()),
                    'box': filtered_boxes[orig_idx].cpu().tolist(),
                    'distance': distance,
                    'urgency': self._get_urgency(class_name),
                    'is_text': bool(filtered_text[orig_idx].item() > 0.5)
                })
            detections.append(img_detections)
        return detections
    
    def _nms(self, boxes: torch.Tensor, scores: torch.Tensor, threshold: float) -> List[int]:
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
                    # Compute IoU between current box and all remaining boxes at once
                    current_box = boxes_corners[idx:idx+1]  # Keep as [1, 4] for broadcasting
                    ious = self._compute_iou_corners(current_box, remaining_boxes)
                    
                    # Suppress boxes that overlap too much (IoU >= threshold)
                    # Higher threshold = more aggressive suppression
                    suppress_mask = ious.flatten() >= threshold
                    suppressed[remaining_idx[suppress_mask]] = True
        
        return keep
    
    def _center_to_corners(self, boxes: torch.Tensor) -> torch.Tensor:
        
        #Convert boxes from center format to corner format which is IoU
        
        x_center, y_center, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x_center - w / 2  # Left edge
        y1 = y_center - h / 2  # Top edge
        x2 = x_center + w / 2  # Right edge
        y2 = y_center + h / 2  # Bottom edge
        return torch.stack([x1, y1, x2, y2], dim=1)
    
    def _compute_iou(self, box1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """
        Compute Intersection over Union IoU between box1 (center format) and all boxes2 (center format)
        
        Args:
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
        
        #Compute IoU between box1 and all boxes2
        
        # Make sure box1 is 2D
        if box1.dim() == 1:
            box1 = box1.unsqueeze(0)
        
        box1 = box1.unsqueeze(1) 
        boxes2 = boxes2.unsqueeze(0) 
        
        # Find the intersection rectangle
        # Two boxes overlap if their intersection exists
        # Top-left corner: max of the two top-left corners (rightmost left, bottommost top)
        inter_x1 = torch.max(box1[..., 0], boxes2[..., 0])
        inter_y1 = torch.max(box1[..., 1], boxes2[..., 1]) 
        # Bottom-right corner: min of the two bottom-right corners (leftmost right, topmost bottom)
        inter_x2 = torch.min(box1[..., 2], boxes2[..., 2])
        inter_y2 = torch.min(box1[..., 3], boxes2[..., 3])
        
        # Calculate intersection area (clamp to 0 in case boxes don't overlap)
        # If boxes don't overlap, inter_x2 < inter_x1, so we clamp to 0
        inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
        inter_area = inter_w * inter_h  # Area of intersection
        
        box1_area = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        
        union_area = box1_area + boxes2_area - inter_area
        
        # 1e-6 is small enough to not affect results but big enough to prevent NaN
        iou = inter_area / (union_area + 1e-6)
        
        if iou.size(0) == 1:
            iou = iou.squeeze(0)
        
        return iou
    
    def _get_urgency(self, class_name: str) -> int:
        #Urgency mapping
        if not hasattr(self, '_urgency_map'):
            self._urgency_map = {
                # Level 3: Danger marked by immediate hazards
                'danger': {
                    'car', 'truck', 'bus', 'motorcycle', 'vehicle', 'traffic',
                    'stairs', 'staircase', 'stairway', 'escalator', 'elevator',
                    'fire', 'emergency', 'hazard', 'construction', 'obstacle'
                },
                # Level 2: Warning marked by requires attention
                'warning': {
                    'bicycle', 'person', 'stop_sign', 'traffic_light', 'crosswalk',
                    'pedestrian', 'yield', 'caution', 'warning'
                },
                # Level 1: Caution marked by moderate importance
                'caution': {
                    'door', 'chair', 'table', 'furniture', 'barrier', 'fence',
                    'wall', 'corner', 'step', 'curb', 'ramp'
                }
            }
        
        class_lower = class_name.lower()
        
        if any(keyword in class_lower for keyword in self._urgency_map['danger']):
            return 3  

        if any(keyword in class_lower for keyword in self._urgency_map['warning']):
            return 2 

        if any(keyword in class_lower for keyword in self._urgency_map['caution']):
            return 1  
        

        return 0 


def create_model(
    num_classes: int = len(COCO_CLASSES),
    condition_mode: Optional[str] = None,
    use_audio: bool = True,
    fpn_channels: int = 256
) -> MaxSightCNN:
    """
    Convenience function to create a MaxSight model
    
    Just wraps the constructor with sensible defaults. Most of the time
    you'll use this instead of calling MaxSightCNN directly.
    
    This is a factory function - makes it easier to create models with
    different configurations without remembering all the default values.
    """
    return MaxSightCNN(
        num_classes=num_classes,
        num_urgency_levels=4,
        num_distance_zones=3,
        use_audio=use_audio,
        condition_mode=condition_mode, 
        fpn_channels=fpn_channels 
    )


# Test model initialization
if __name__ == "__main__":
    import time
    
    print("MaxSight CNN")
    
    # Test 1: Basic inference
    print("\n Test 1: Basic Inference")
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
    print("\n Test 2: Detection Post-Processing")
    detections = model.get_detections(outputs, confidence_threshold=0.3)
    print(f"  Image 1: {len(detections[0])} detections")
    print(f"  Image 2: {len(detections[1])} detections")
    
    if len(detections[0]) > 0:
        print(f"  Sample detection: {detections[0][0]}")
    
    # Test 3: NMS functionality
    print("\n Test 3: NMS Verification")
    test_boxes = torch.tensor([
        [0.5, 0.5, 0.2, 0.2],
        [0.52, 0.52, 0.2, 0.2],  # High overlap
        [0.8, 0.8, 0.2, 0.2],     # Low overlap
    ])
    test_scores = torch.tensor([0.9, 0.8, 0.7])
    keep = model._nms(test_boxes, test_scores, threshold=0.5)
    print(f"  Input boxes: {len(test_boxes)}, Kept after NMS: {len(keep)}")
    
    # Test 4: IoU computation
    print("\n Test 4: IoU Computation")
    box1 = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    box2 = torch.tensor([[0.52, 0.52, 0.2, 0.2], [0.8, 0.8, 0.2, 0.2]])
    ious = model._compute_iou(box1, box2)
    print(f"  IoU scores: {ious.squeeze().tolist()}")
    
    # Test 5: Urgency mapping
    print("\n Test 5: Urgency Mapping")
    test_classes = ['car', 'person', 'door', 'vase']
    for cls in test_classes:
        urgency = model._get_urgency(cls)
        print(f"  {cls}: urgency level {urgency}")
    
    # Test 6: Model size
    print("\n Test 6: Model Size Check")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  FP32 size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    print(f"  INT8 size: ~{total_params / 1024 / 1024:.1f} MB")
    print(f"  Target <50MB: {'good to go😄' if total_params / 1024 / 1024 < 50 else 'no'}")
    
    # Test 7: Inference timing
    print("\n Test 7: Inference Latency")
    times = []
    with torch.no_grad():
        for i in range(20):
            start = time.time()
            i = model(dummy_image)
            times.append((time.time() - start) * 1000)
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    print(f"  Average: {avg_time:.1f}ms")
    print(f"  Min: {min_time:.1f}ms, Max: {max_time:.1f}ms")
    print(f"  Target <500ms: {'good to go😄' if avg_time < 500 else '✗'}")
    
    # Test 8: Condition-specific modes
    print("\n Test 8: Condition-Specific Modes")
    for condition in ['glaucoma', 'amd', 'color_blindness']:
        cond_model = create_model(condition_mode=condition)
        with torch.no_grad():
            cond_outputs = cond_model(dummy_image)
        print(f"  {condition}: {len(cond_outputs)} outputs")
    
    
    print("Model ready for deployment")
