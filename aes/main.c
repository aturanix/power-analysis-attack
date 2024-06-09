#include "aes.h"

#include <stdio.h>
#include <stdint.h>

int main(void)
{
    uint8_t key[] = "Thats my Kung Fu";
    uint8_t in[] = "Two One Nine Two";
    uint8_t c[16];
    aes128_cipher(in, key, c);
    for (size_t i = 0; i != 16; ++i) {
        printf("%x", c[i]);
    }
    printf("\n");
}

