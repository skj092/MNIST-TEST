from torchvision.datasets import MNIST
import torch, torchvision
from model import Net
import CFG
from torch.utils.data import DataLoader
from CFG import transform
from model import LeNet
from CFG import load_checkpoint


# evaluate
def evaluate(model, test_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(CFG.device)
            images = images.view(images.size(0), 1, 28, 28)
            # images = images.reshape(images.shape[0], -1)
            labels = labels.to(CFG.device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

if __name__=="__main__":
    # loading test MNIST dataset
    image_path = '.'
    mnist_test_ds = MNIST(root=image_path, train=False, transform=transform, download=True)
    test_loader = DataLoader(mnist_test_ds, batch_size=CFG.BATCH_SIZE)

    # loading model
    model = LeNet()
    load_checkpoint(torch.load(CFG.CHECKPOINT_FILE), model)
    print('model loaded successfully')
    model.to(CFG.device)

    evaluate(model, test_loader)