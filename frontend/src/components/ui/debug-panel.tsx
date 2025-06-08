"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

interface DebugPanelProps {
  apiBase: string
  authToken: string | null
  user: any
}

export function DebugPanel({ apiBase, authToken, user }: DebugPanelProps) {
  const [isOpen, setIsOpen] = useState(false)
  const [apiStatus, setApiStatus] = useState<Record<string, string>>({})

  const checkApiEndpoint = async (endpoint: string) => {
    try {
      const response = await fetch(`${apiBase}${endpoint}`, {
        headers: authToken ? { Authorization: `Bearer ${authToken}` } : {},
      })
      setApiStatus((prev) => ({
        ...prev,
        [endpoint]: response.ok ? "✅ OK" : `❌ ${response.status}`,
      }))
    } catch (error) {
      setApiStatus((prev) => ({
        ...prev,
        [endpoint]: "❌ Error",
      }))
    }
  }

  const testEndpoints = ["/", "/api/voices", "/api/me"]

  if (!isOpen) {
    return (
      <Button onClick={() => setIsOpen(true)} variant="outline" size="sm" className="fixed bottom-4 right-4 z-50">
        Debug
      </Button>
    )
  }

  return (
    <Card className="fixed bottom-4 right-4 w-80 z-50">
      <CardHeader>
        <CardTitle className="flex justify-between items-center">
          Debug Panel
          <Button onClick={() => setIsOpen(false)} variant="ghost" size="sm">
            ×
          </Button>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        <div>
          <strong>API Base:</strong> {apiBase}
        </div>
        <div>
          <strong>Auth Token:</strong>
        </div>
        <div>
          <strong>User:</strong>
        </div>

        <div className="space-y-1">
          <strong>API Status:</strong>
          {testEndpoints.map((endpoint) => (
            <div key={endpoint} className="flex justify-between">
              <span className="text-sm">{endpoint}</span>
              <div className="flex gap-1">
                <span className="text-xs">{apiStatus[endpoint] || "?"}</span>
                <Button onClick={() => checkApiEndpoint(endpoint)} size="sm" variant="ghost" className="h-4 w-4 p-0">
                  ↻
                </Button>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
