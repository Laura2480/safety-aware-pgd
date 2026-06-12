
import open_clip
from sklearn.metrics.pairwise import cosine_similarity
import torch
from src.pgd_safety_aware.moderation_clip import ModerationModel


class MessageExtractionError(Exception):
    pass

class ClipDetector:
    def __init__(self, checkpoint_path,client=None, device=None, is_big_g=False):
        self.moderation_count = 0

        if device is not None:
            self.device = device
        else:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.clip_pretrain = "laion2b_s39b_b160k" if is_big_g else "laion2b_s32b_b82k"
        self.clip_model = "ViT-bigG-14" if is_big_g else "ViT-L-14"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(self.clip_model, pretrained=self.clip_pretrain,
                                                                               device=self.device)
        self.moderation_model = ModerationModel(self.clip_model)
        self.moderation_model.load_state_dict(torch.load(checkpoint_path))
        self.moderation_model = self.moderation_model.to(self.device)
        self.tokenizer = open_clip.get_tokenizer(self.clip_model)
        #self.model.eval()
        self.labels=['harassment', 'harassment_threatening', 'hate',
    'hate_threatening', 'self_harm', 'self_harm_instructions',
    'self_harm_intent', 'sexual', 'sexual_minors',
    'violence', 'violence_graphic']


    # Safe word filtering


    def measure_text_similarity(self, prompt, text_list):
        # Tokenize the prompt
        prompt_input = self.tokenizer([prompt]).to(self.device)

        # Tokenize the text_list
        text_inputs = self.tokenizer(text_list).to(self.device)

        # Get text features for the prompt
        with torch.no_grad():
            prompt_features = self.model.encode_text(prompt_input)

        # Normalize prompt features
        prompt_features = prompt_features / prompt_features.norm(dim=-1, keepdim=True)

        # Get text features for the text_list
        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs)

        # Normalize text features
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Compute cosine similarity between the prompt and each item in text_list
        similarity_scores = cosine_similarity(text_features.cpu().numpy(), prompt_features.cpu().numpy()).T[0]

        return similarity_scores


    # Test inversion


    def get_target_feature(self, target_images=None, target_text=None):
        """
        Retrieves target features from images or text prompts.
        """
        with torch.no_grad():
            if target_images is not None:
                curr_images = [self.preprocess(i).unsqueeze(0) for i in target_images]
                curr_images = torch.cat(curr_images).to(self.device)
                all_target_features = self.model.encode_image(curr_images)
            else:
                texts = self.tokenizer(target_text).to(self.device)
                all_target_features = self.model.encode_text(texts)

        return all_target_features

    def moderate(self, text=None, images=None):
        assert (text is not None and images is None) or (text is None and images is not None)
        scores = None
        if text is not None:
            features = self.get_target_feature(target_text=text)
            scores = torch.sigmoid(self.moderation_model(features))
        if images is not None:
            features = self.get_target_feature(target_images=images)
            scores = torch.sigmoid(self.moderation_model(features))
        val = scores[0].to('cpu').detach().numpy().sum()
        if val > 0.1:
            return True
        else:
            return False

    def measure_similarity(self, text, image):

        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
        text_tokens = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        similarity = cosine_similarity(image_features.cpu().numpy(), text_features.cpu().numpy())

        return similarity[0][0]

