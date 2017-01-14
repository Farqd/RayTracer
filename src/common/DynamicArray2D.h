#ifndef COMMON_DYNAMIC2DARRAY_H
#define COMMON_DYNAMIC2DARRAY_H

#include <vector>

template <typename T>
struct DynamicArray2D
{
  DynamicArray2D(std::size_t rows, std::size_t cols)
    : rows(rows)
    , cols(cols)
    , data_(rows * cols)
  {
  }

  T operator()(std::size_t i, std::size_t j) const
  {
    return data_[i * cols + j];
  }

  T& operator()(std::size_t i, std::size_t j)
  {
    return data_[i * cols + j];
  }

  T* data()
  {
    return data_.data();
  }
  std::size_t size() const
  {
    return data_.size();
  }

  std::size_t const rows;
  std::size_t const cols;

private:
  std::vector<T> data_;
};

#endif // COMMON_DYNAMIC2DARRAY_H
