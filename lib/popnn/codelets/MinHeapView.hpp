#include <cstddef>
#include <utility>

namespace popnn {

/*
  MinHeapView takes a reference to container type and treats it like a heap. The
  container is assumed to be allocated with the current size being given by a
  parameter to the push/replace functions. This is designed to be used by the
  topK codelets which want to replace the smallest value at the top so we can
  replace inplace by removing the 0th element (the smallest) and then rotate
  down to repair the tree.
 */
template <typename IndexVectorType, typename DataVectorType, typename DataType>
class MinHeapView {
public:
  MinHeapView(IndexVectorType &vec, DataVectorType &underlayingStorage)
      : partialBucket(vec), resourceVector(underlayingStorage) {}

  // Reference to the underlaying storage which we are treating as a min heap.
  IndexVectorType &partialBucket;

  DataVectorType &resourceVector;

  // Return the parent of the binary heap.
  inline int GetParent(int i) const { return (i - 1) / 2; }

  // Returns true if this value is larger than the smallest node in the heap.
  inline bool IsLargerThanSmallest(DataType newVal) const {
    return newVal > resourceVector[partialBucket[0]];
  }

  void ReplaceValue(DataType value, int ind, const size_t size) {
    int index = ind;
    partialBucket[index] = value;

    if (index == 0)
      return;
    // For the worst log(n) case with early exit.
    std::size_t parentIndex = index;
    do {
      parentIndex = GetParent(index);

      // If we are in the correct position in the tree, exit.
      if (resourceVector[partialBucket[parentIndex]] < resourceVector[value])
        break;

      // Otherwise we should continue rotating up.
      // Swap the values.
      partialBucket[index] = partialBucket[parentIndex];
      partialBucket[parentIndex] = value;
      index = parentIndex;
    } while (parentIndex != 0);
  }

  // Push to the binary heap by rotating up the values.
  inline void Push(DataType value, const size_t size) {
    // Since the array has been preallocated we can push by "replacing" the
    // value at the end of the logical size. Should save on code.
    ReplaceValue(value, size, size + 1);
  }

  // Pop a value from the binary heap.
  DataType Pop(const size_t size) {
    if (size == 0) {
      return partialBucket[0];
    }
    const size_t newSize = size - 1;

    DataType valueToReturn = partialBucket[0];

    // Swap the smallest element at the top for the element at the bottom.
    std::swap(partialBucket[0], partialBucket[newSize]);

    // Repair the tree now we have broken the heap condition.
    RepairTree(newSize);
    return valueToReturn;
  }

  // Replace the smallest value in the heap and then repair the heap.
  void ReplaceAndRepair(DataType value, const size_t size) {
    if (size == 0) {
      partialBucket[0] = value;
      return;
    }

    // Replace the smallest element.
    partialBucket[0] = value;

    // Repair the tree now we have (maybe) broken the heap condition.
    RepairTree(size);
  }

  void RepairTree(const size_t size, const size_t offset = 0) {
    int index = 0;
    // For the worst log(n) case with early exit.
    do {

      std::size_t left = 2 * index + 1;
      std::size_t right = 2 * index + 2;
      bool largerThanLeft = left < size
                                ? resourceVector[partialBucket[index]] >
                                      resourceVector[partialBucket[left]]
                                : false;
      bool largerThanRight = right < size
                                 ? resourceVector[partialBucket[index]] >
                                       resourceVector[partialBucket[right]]
                                 : false;

      // If we are smaller than both our children we are in the right place.
      if (!(largerThanLeft || largerThanRight)) {
        break;
      }

      // Otherwise we should continue rotating down. Swap the right unless we
      // can swap the left and it is smaller than the right.
      if (largerThanRight &&
          !(largerThanLeft && resourceVector[partialBucket[right]] >
                                  resourceVector[partialBucket[left]])) {
        std::swap(partialBucket[index], partialBucket[right]);
        index = right;
      } else if (largerThanLeft) {
        std::swap(partialBucket[index], partialBucket[left]);
        index = left;
      }
    } while (index < size);
  }

  void Sort(size_t size) {
    // Pop each element off. This involves moving the smallest (the top) to the
    // back of the array.
    for (int i = size; i >= 0; --i) {
      this->Pop(i);
    }
  }
};

} // namespace popnn
