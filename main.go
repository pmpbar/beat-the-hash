package main

import (
	"encoding/hex"
	"flag"
	"github.com/pmpbar/log"
	"github.com/whyrusleeping/FastGoSkein"
	"math/rand"
)

var l logger.Logger

func main() {
	var routines = flag.Int("r", 4, "Amount of go routines to start")
	flag.Parse()

	l = logger.NewLogger(logger.LEVELDEBUG)

	bestDistance := 450

	hdc := make(chan int)
	input := make(chan []byte)

	l.Debug("Starting %d routines", *routines)
	for i := 0; i < *routines; i++ {
		go Guess(hdc, input)
	}

	l.Info("Hashin away...")
	for distance := range hdc {
		in := <-input
		if distance < bestDistance {
			l.Info("New best Distance %d", distance)
			l.Info("Input: %v", hex.EncodeToString(in))
			bestDistance = distance
		}
	}
}
func Guess(hdc chan int, input chan []byte) {
	rmh := RandellMunroeHash()
	bestDistance := 450
	toHash := make([]byte, 32)
	for {
		rand.Read(toHash)
		skHash := Hash(toHash)
		distance := Hamming(rmh, skHash)

		if distance < bestDistance {
			l.Verbose("Hash: %v", toHash)
			bestDistance = distance
			hdc <- distance
			input <- toHash
		}
	}
}

func Hash(toHash []byte) []byte {
	sk := new(skein.Skein1024)
	sk.Init(1024)
	sk.Update(toHash)
	outputBuffer := make([]byte, 128)
	sk.Final(outputBuffer)
	return outputBuffer[:]
}

func Hamming(target, output []byte) int {
	count := 0
	for i, b1 := range target {
		b2 := output[i]
		for x := b1 ^ b2; x > 0; x >>= 1 {
			if int(x&1) == 1 {
				count++
			}
		}
	}
	if count == 0 {
		return 1
	}
	return count
}

func RandellMunroeHash() []byte {
	return []byte{0x5b, 0x4d, 0xa9, 0x5f, 0x5f, 0xa0, 0x82, 0x80,
		0xfc, 0x98, 0x79, 0xdf, 0x44, 0xf4, 0x18, 0xc8, 0xf9, 0xf1,
		0x2b, 0xa4, 0x24, 0xb7, 0x75, 0x7d, 0xe0, 0x2b, 0xbd, 0xfb,
		0xae, 0x0d, 0x4c, 0x4f, 0xdf, 0x93, 0x17, 0xc8, 0x0c, 0xc5,
		0xfe, 0x04, 0xc6, 0x42, 0x90, 0x73, 0x46, 0x6c, 0xf2, 0x97,
		0x06, 0xb8, 0xc2, 0x59, 0x99, 0xdd, 0xd2, 0xf6, 0x54, 0x0d,
		0x44, 0x75, 0xcc, 0x97, 0x7b, 0x87, 0xf4, 0x75, 0x7b, 0xe0,
		0x23, 0xf1, 0x9b, 0x8f, 0x40, 0x35, 0xd7, 0x72, 0x28, 0x86,
		0xb7, 0x88, 0x69, 0x82, 0x6d, 0xe9, 0x16, 0xa7, 0x9c, 0xf9,
		0xc9, 0x4c, 0xc7, 0x9c, 0xd4, 0x34, 0x7d, 0x24, 0xb5, 0x67,
		0xaa, 0x3e, 0x23, 0x90, 0xa5, 0x73, 0xa3, 0x73, 0xa4, 0x8a,
		0x5e, 0x67, 0x66, 0x40, 0xc7, 0x9c, 0xc7, 0x01, 0x97, 0xe1,
		0xc5, 0xe7, 0xf9, 0x02, 0xfb, 0x53, 0xca, 0x18, 0x58, 0xb6}
}