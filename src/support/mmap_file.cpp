// MoEx clean-room. Windows read-only file mapping implementation.
#include "support/mmap_file.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

namespace moex {

bool MmapFile::open(const std::string& path, std::string* err) {
  close_();
  HANDLE fh = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                          OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (fh == INVALID_HANDLE_VALUE) {
    if (err) *err = "cannot open '" + path + "' (win32 err " +
                    std::to_string(GetLastError()) + ")";
    return false;
  }
  LARGE_INTEGER li;
  if (!GetFileSizeEx(fh, &li)) {
    if (err) *err = "GetFileSizeEx failed for '" + path + "'";
    CloseHandle(fh);
    return false;
  }
  uint64_t sz = static_cast<uint64_t>(li.QuadPart);
  if (sz == 0) {
    if (err) *err = "empty file '" + path + "'";
    CloseHandle(fh);
    return false;
  }
  HANDLE mh = CreateFileMappingA(fh, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (!mh) {
    if (err) *err = "CreateFileMapping failed (win32 err " +
                    std::to_string(GetLastError()) + ")";
    CloseHandle(fh);
    return false;
  }
  const void* base = MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0);
  if (!base) {
    if (err) *err = "MapViewOfFile failed (win32 err " +
                    std::to_string(GetLastError()) + ")";
    CloseHandle(mh);
    CloseHandle(fh);
    return false;
  }
  file_ = fh;
  mapping_ = mh;
  data_ = static_cast<const uint8_t*>(base);
  size_ = sz;
  return true;
}

void MmapFile::close_() {
  if (data_) { UnmapViewOfFile(data_); data_ = nullptr; }
  if (mapping_) { CloseHandle(static_cast<HANDLE>(mapping_)); mapping_ = nullptr; }
  if (file_) { CloseHandle(static_cast<HANDLE>(file_)); file_ = nullptr; }
  size_ = 0;
}

MmapFile::~MmapFile() { close_(); }

MmapFile::MmapFile(MmapFile&& o) noexcept
    : file_(o.file_), mapping_(o.mapping_), data_(o.data_), size_(o.size_) {
  o.file_ = o.mapping_ = nullptr;
  o.data_ = nullptr;
  o.size_ = 0;
}

MmapFile& MmapFile::operator=(MmapFile&& o) noexcept {
  if (this != &o) {
    close_();
    file_ = o.file_; mapping_ = o.mapping_; data_ = o.data_; size_ = o.size_;
    o.file_ = o.mapping_ = nullptr; o.data_ = nullptr; o.size_ = 0;
  }
  return *this;
}

}  // namespace moex
