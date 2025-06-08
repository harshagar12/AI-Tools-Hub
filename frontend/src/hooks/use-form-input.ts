"use client"

import type React from "react"

import { useState, useCallback } from "react"

type InputChangeHandler = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => void

interface FormInputHook<T> {
  values: T
  setValue: (name: keyof T, value: any) => void
  handleChange: InputChangeHandler
  reset: () => void
}

export function useFormInput<T extends Record<string, any>>(initialValues: T): FormInputHook<T> {
  const [values, setValues] = useState<T>(initialValues)

  const setValue = useCallback((name: keyof T, value: any) => {
    setValues((prev) => ({ ...prev, [name]: value }))
  }, [])

  const handleChange: InputChangeHandler = useCallback(
    (e) => {
      const { name, value } = e.target
      setValue(name, value)
    },
    [setValue],
  )

  const reset = useCallback(() => {
    setValues(initialValues)
  }, [initialValues])

  return { values, setValue, handleChange, reset }
}
