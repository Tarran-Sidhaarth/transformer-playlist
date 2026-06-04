class Schema:
    def __init__(self,name:str, weight: float, price: float=0) -> None:
        self.name = name
        self.weight = weight
        self.price = price

def get_distance(new_point: Schema,data_points: list[Schema]) -> list:
    """
    Computes the absolute weight distance between a new point and each data point.

    Args:
        new_point:    The query Schema whose price we want to estimate.
        data_points:  List of reference Schema objects with known prices.

    Returns:
        A list of floats where each element is abs(new_point.weight - point.weight).
    """

def get_similarity(distances: list) -> list:
    """
    Converts distances to similarity scores using the inverse distance formula.

    Args:
        distances:  List of non-zero float distances from get_distance().

    Returns:
        A list of floats where each element is 1 / distance.
    """
    

def get_coefficients(similarity_score:list) -> list:
    """
    Normalises similarity scores into weights that sum to 1.

    Args:
        similarity_score:  List of raw similarity floats from get_similarity().

    Returns:
        A list of floats (coefficients) where each element is
        similarity / sum(similarity_score).
    """

def get_weighted_price(new_point: Schema, data_points: list[Schema]):
    """
    Estimates the price of a new point via inverse-distance weighted averaging.

    Args:
        new_point:    The query Schema whose price we want to estimate.
        data_points:  List of reference Schema objects with known prices.

    Returns:
        A float representing the weighted average price across all data points.
    """

















# a = Schema("1",500,50)
# b = Schema("2",600, 60)
# c= Schema("3",700,70)

# print(get_weighted_price(Schema("3",560,0),[a,b]))



    
        

