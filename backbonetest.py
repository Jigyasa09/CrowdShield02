# test_backbone.py
import os
import torch
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_backbone_file():
    """Check if backbone file exists and can be loaded"""
    backbone_path = 'checkpoints/vgg16_bn-6c64b313.pth'
    
    print(f"Checking for backbone at: {backbone_path}")
    
    if not os.path.exists(backbone_path):
        print(f"❌ Backbone file not found at: {backbone_path}")
        if os.path.exists('checkpoints'):
            print(f"Files in checkpoints: {os.listdir('checkpoints')}")
        return False
    
    print(f"✅ Backbone file found")
    
    try:
        backbone_state_dict = torch.load(backbone_path, map_location='cpu')
        print(f"✅ Backbone loaded successfully")
        print(f"   Number of keys: {len(backbone_state_dict)}")
        print(f"   Sample keys: {list(backbone_state_dict.keys())[:5]}")
        return True
    except Exception as e:
        print(f"❌ Failed to load backbone: {e}")
        return False

def test_vgg_model():
    """Test if we can create a VGG model with the backbone"""
    try:
        import torchvision.models as models
        vgg = models.vgg16_bn(pretrained=False)
        print(f"✅ VGG16_BN model created successfully")
        
        # Try to load our backbone
        backbone_path = 'checkpoints/vgg16_bn-6c64b313.pth'
        if os.path.exists(backbone_path):
            state_dict = torch.load(backbone_path, map_location='cpu')
            vgg.load_state_dict(state_dict, strict=False)
            print(f"✅ Backbone weights loaded into VGG model")
        return True
    except Exception as e:
        print(f"❌ VGG model test failed: {e}")
        return False

if __name__ == '__main__':
    print("="*50)
    print("Testing Backbone Loading")
    print("="*50)
    
    test_backbone_file()
    print()
    test_vgg_model()