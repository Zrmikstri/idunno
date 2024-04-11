import torch

from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.feature_extraction import (
    get_graph_node_names,
    create_feature_extractor
)

model = resnet50(weights='DEFAULT')

# large feature maps
## make the features before gap 14 x 14
model.layer3[0].conv2.stride = (1, 1)
model.layer3[0].downsample[0].stride = (1, 1)
## make the features before gap 28 x 28
model.layer4[0].conv2.stride = (1, 1)
model.layer4[0].downsample[0].stride = (1, 1)

preprocess = ResNet50_Weights.DEFAULT.transforms()

train_nodes, eval_nodes = get_graph_node_names(model)

return_nodes = {
    'layer4.2.relu_2': 'resnet50_features',
}

feature_extractor = create_feature_extractor(model, return_nodes)

inp = torch.randn(5, 3, 224, 224)
inp = preprocess(inp)

with torch.no_grad():
    features = feature_extractor(inp)['resnet50_features']
    
    weight = model.fc.weight

# features     :     5 x 2048 x 28 x 28
# weights      :  1000 x 2048
# Result       :     5 x 1000 x 28 x 28

result = torch.einsum('bchw,ic->bihw', features, weight)

print(result.shape)
