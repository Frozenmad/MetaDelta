""" Defines the API used in the Cross-Domain MetaDL competition. Please check 
the dedicated notebook tutorial (cd-metadl/tutorial.ipynb) for details.

AS A PARTICIPANT, DO NOT MODIFY THIS CODE
"""
import numpy as np


class Predictor():
    """ This class represents the predictor returned at the end of the 
    Learner's fit method. 
    """
    
    def __init__(self) -> None:
        """ Defines the Predictor initialization.
        """
        pass

    def predict(self, query_set) -> np.ndarray:
        """ Given a query_set, predicts the probabilities associated to the 
        provided images or the labels to the provided images.
        
        Args:
            query_set (Tensor): Tensor of unlabelled image examples of shape 
                [n_ways*query_size x 3 x 128 x 128].
        
        Returns:
            np.ndarray: It can be:
                - Raw logits matrix (the logits are the unnormalized final 
                    scores of your model). The matrix must be of shape 
                    [n_ways*query_size, n_ways]. 
                - Predicted label probabilities matrix. The matrix must be of 
                    shape [n_ways*query_size, n_ways].
                - Predicted labels. The array must be of shape 
                    [n_ways*query_size].
        """
        raise NotImplementedError(("You should implement the predict method "
            + "for the Predictor class."))
 

class Learner():
    """ This class represents the learner returned at the end of the 
    meta-learning procedure.
    """
    
    def __init__(self) -> None:
        """ Defines the learner initialization.
        """
        pass

    def fit(self, support_set) -> Predictor:
        """ Fit the Learner to the support set of a new unseen task. 
        
        Args:
            support_set (Tuple[Tensor, Tensor, Tensor, int, int]): Support set 
                of a task. The data arrive in the following format (X_train, 
                y_train, original_y_train, n_ways, k_shots). X_train is the 
                tensor of labeled images of shape [n_ways*k_shots x 3 x 128 x 
                128], y_train is the tensor of encoded labels (Long) for each 
                image in X_train with shape of [n_ways*k_shots], 
                original_y_train is the tensor of original labels (Long) for 
                each image in X_train with shape of [n_ways*k_shots], n_ways is
                the number of classes and k_shots the number of examples per 
                class.
                        
        Returns:
            Predictor: The resulting predictor ready to predict unlabelled 
                query image examples from new unseen tasks.
        """
        raise NotImplementedError(("You should implement the fit method for "
            + "the Learner class."))
    
    def save(self, path_to_save) -> None:
        """ Saves the learning object associated to the Learner. 
        
        Args:
            path_to_save (str): Path where the learning object will be saved.

        Note: It is mandatory to allow saving the Learner as a file(s) in 
        path_to_save. Otherwise, it won't be a valid submission.
        """
        raise NotImplementedError(("You should implement the save method for "
            + "the Learner class."))

    def load(self, path_to_load) -> None:
        """ Loads the learning object associated to the Learner. It should 
        match the way you saved this object in self.save().
        
        Args:
            path_to_load (str): Path where the Learner is saved.
        """
        raise NotImplementedError(("You should implement the load method for "
            + "the Learner class."))

       
class MetaLearner():
    """ Define the meta-learning algorithm we want to use, through its methods.
    It is an abstract class so one has to overide the core methods depending 
    on the algorithm.
    """
    
    def __init__(self, 
                 train_classes, 
                 total_classes, 
                 logger) -> None:
        """ Defines the meta-learning algorithm's parameters. For example, one 
        has to define what would be the meta-learner's architecture. 
        
        Args:
            train_classes (int): Total number of classes that can be seen 
                during meta-training. If the data format during training is 
                'task', then this parameter corresponds to the number of ways, 
                while if the data format is 'batch', this parameter corresponds 
                to the total number of classes across all training datasets.
            total_classes (int): Total number of classes across all training 
                datasets. If the data format during training is 'batch' this 
                parameter is exactly the same as train_classes.
            logger (Logger): Logger that you can use during meta-learning 
                (HIGHLY RECOMMENDED). You can use it after each meta-train or 
                meta-validation iteration as follows: 
                    self.log(data, predictions, loss, meta_train)
                - data (task or batch): It is the data used in the current 
                    iteration.
                - predictions (np.ndarray): Predictions associated to each test 
                    example in the specified data. It can be the raw logits 
                    matrix (the logits are the unnormalized final scores of 
                    your model), a probability matrix, or the predicted labels.
                - loss (float, optional): Loss of the current iteration. 
                    Defaults to None.
                - meta_train (bool, optional): Boolean flag to control if the 
                    current iteration belongs to meta-training. Defaults to 
                    True.
        """
        self.train_classes = train_classes
        self.total_classes = total_classes
        self.log = logger.log

    def meta_fit(self, 
                 meta_train_generator, 
                 meta_valid_generator) -> Learner:
        """ Uses the generators to tune the meta-learner's parameters. The 
        meta-training generator generates either few-shot learning tasks or 
        batches of images, while the meta-valid generator always generates 
        few-shot learning tasks.
        
        Args:
            meta_train_generator (Iterable[Any]): Function that generates the 
                training data. The generated can be a N-way k-shot task or a 
                batch of images with labels.
            meta_valid_generator (Iterable[Task]): Function that generates the 
                validation data. The generated data always come in form of 
                N-way k-shot tasks.
                
        Returns:
            Learner: Resulting learner ready to be trained and evaluated on new
                unseen tasks.
                
        Note: 
        Each N-way k-shot task is an object with the following attributes: 
            num_ways (int): Number of ways (classes) in the current task. 
            num_shots (int): Number of shots (images per class) for the support 
                set.
            support_set (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): 
                Support set for the current task. The first tensor corresponds 
                to the images with a shape of [num_ways*num_shots x 3 x 128 x 
                128]. The second tensor corresponds to the labels with a shape 
                of [num_ways*num_shots].The last tensor corresponds to the 
                original labels with a shape of [num_ways*num_shots].
            query_set (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Query 
                set for the current task. The first tensor corresponds to the 
                images with a shape of [num_ways*query_size x 3 x 128 x 128]. 
                The second tensor corresponds to the labels with a shape of 
                [num_ways*query_size] and the last tensor corresponds to the 
                original labels with a shape of [num_ways*num_shots]. The 
                query_size can vary depending on the configuration of the data 
                loader.
            original_class_idx (np.ndarray): Array with the original class 
                indexes used in the current task, its shape is [num_ways, ].
            dataset (str): Name of the dataset used to create the current task. 
                 
        On the other hand each batch is composed of images and labels in the 
        following format: List[torch.Tensor, torch.Tensor]. The first tensor 
        corresponds to the images with a shape of [batch_size x 3 x 128 x 128] 
        while the second array corresponds to the labels with a shape of 
        [batch_size].
        """
        raise NotImplementedError(("You should implement the meta_fit method "
            + f"for the MetaLearner class."))
