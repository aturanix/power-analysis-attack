#pragma once

#include <stdint.h>
#include <stddef.h>

void aes128_cipher(uint8_t const *in, uint8_t const *key, uint8_t *out);
