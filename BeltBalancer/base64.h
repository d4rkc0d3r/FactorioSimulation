#ifndef BASE64_H_INCLUDED
#define BASE64_H_INCLUDED

#include<string>

std::string base64_encode(unsigned char const* bytes_to_encode, unsigned int in_len);
std::string base64_decode(std::string const& encoded_string);

#endif // BASE64_H_INCLUDED
